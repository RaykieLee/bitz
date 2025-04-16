use std::{
    io::stdout,
    sync::{Arc},
    sync::atomic::{AtomicU32, Ordering},
    thread::sleep,
    time::{Duration, Instant},
    usize,
};

use b64::FromBase64;
use colored::*;
use crossterm::{
    cursor::MoveTo,
    execute,
    terminal::{Clear, ClearType},
};
use drillx::{
    equix::{self},
    Hash, Solution,
};
use eore_api::{
    consts::{BUS_ADDRESSES, BUS_COUNT, EPOCH_DURATION},
    event::MineEvent,
    state::{proof_pda, Bus, Config},
};
use rand::Rng;
use solana_program::pubkey::Pubkey;
use solana_rpc_client::spinner;
use solana_sdk::{signature::Signature, signer::Signer};
use solana_transaction_status::{option_serializer::OptionSerializer, UiTransactionEncoding};
use steel::AccountDeserialize;
use tabled::{
    settings::{
        object::{Columns, Rows},
        style::BorderColor,
        Alignment, Border, Color, Highlight, Remove, Style,
    },
    Table,
};

use crate::{
    args::CollectArgs,
    error::Error,
    utils::{
        amount_u64_to_f64, format_duration, format_timestamp, get_clock, get_config,
        get_updated_proof_with_authority, ComputeBudget, PoolCollectingData, SoloCollectingData,
    },
    Miner,
};

#[cfg(feature = "gpu")]
extern "C" {
    pub static BATCH_SIZE: u32;
    pub fn gpu_hash(challenge: *const u8, nonce: *const u8, out: *mut u64);
    pub fn gpu_solve_stages(hashes: *const u64, out: *mut u8, sols: *mut u32, num_sets: i32);
    pub fn gpu_unified_mine(challenge: *const u8, nonce: *const u8, out: *mut u8, sols: *mut u32, num_sets: i32);
}

use super::pool::Pool;
use rayon::prelude::*;

impl Miner {
    pub async fn collect(&self, args: CollectArgs) -> Result<(), Error> {
        // 设置GPU环境变量
        #[cfg(feature = "gpu")]
        if args.gpu {
            std::env::set_var("BITZ_USE_GPU", "1");
            
            // 设置GPU参数环境变量
            std::env::set_var("BITZ_GPU_BATCH_SIZE", args.batch_size.to_string());
            std::env::set_var("BITZ_GPU_HASH_THREADS", args.hash_threads.to_string());
            std::env::set_var("BITZ_GPU_SOLVE_THREADS", args.solve_threads.to_string());
            
            println!("{} Using GPU for collecting (Device: {}, Batch Size: {}, Hash Threads: {}, Solve Threads: {})", 
                "INFO".bold().green(), 
                args.gpu_device,
                args.batch_size,
                args.hash_threads,
                args.solve_threads
            );
        } else {
            std::env::set_var("BITZ_USE_GPU", "0");
        }

        match args.pool_url {
            Some(ref pool_url) => {
                let pool = &Pool {
                    http_client: reqwest::Client::new(),
                    pool_url: pool_url.clone(),
                };
                self.collect_pool(args, pool).await?;
            }
            None => {
                self.collect_solo(args).await;
            }
        }
        Ok(())
    }

    async fn collect_solo(&self, args: CollectArgs) {
        // Open account, if needed.
        self.open().await;

        // Check num threads
        let cores_str = args.cores;
        let cores = if cores_str == "ALL" {
            num_cpus::get() as u64
        } else {
            cores_str.parse::<u64>().unwrap()
        };
        self.check_num_cores(cores);

        // Get verbose flag
        let verbose = args.verbose;

        // Generate addresses
        let signer = self.signer();
        let _proof_address = proof_pda(signer.pubkey()).0;
        let boost_config_address = eore_boost_api::state::config_pda().0;

        // Start collecting loop
        let mut last_hash_at = 0;
        loop {
            // Fetch accounts
            let clock = get_clock(&self.rpc_client)
                .await
                .expect("Failed to fetch clock account");
            let config = get_config(&self.rpc_client).await;
            let proof =
                get_updated_proof_with_authority(&self.rpc_client, signer.pubkey(), last_hash_at)
                    .await
                    .expect("Failed to fetch proof account");

            // Log collecting table
            self.update_solo_collecting_table(verbose);

            // Track timestamp
            last_hash_at = proof.last_hash_at;

            // Calculate cutoff time
            let cutoff_time = self.get_cutoff(&clock, proof.last_hash_at, args.buffer_time);

            // Run drillx
            let solution = Self::find_hash_par(
                proof.challenge,
                cutoff_time,
                cores,
                config.min_difficulty as u32,
                None,
            )
            .await;

            // Build instruction set
            let mut ixs = vec![eore_api::sdk::auth(proof_pda(signer.pubkey()).0)];
            let mut compute_budget = 750_000;

            // Check for reset
            if self.should_reset(&clock, config).await
            // && rand::thread_rng().gen_range(0..100).eq(&0)
            {
                compute_budget += 100_000;
                ixs.push(eore_api::sdk::reset(signer.pubkey()));
            }

            // Build collect ix
            let collect_ix = eore_api::sdk::mine(
                signer.pubkey(),
                signer.pubkey(),
                self.find_bus().await,
                solution,
                boost_config_address,
            );
            ixs.push(collect_ix);

            // Submit transaction
            match self
                .send_and_confirm(&ixs, ComputeBudget::Fixed(compute_budget), false)
                .await
            {
                Ok(sig) => self.fetch_solo_collect_event(sig, verbose).await,
                Err(err) => {
                    let collecting_data = SoloCollectingData::failed();
                    let mut data = self.solo_collecting_data.write().unwrap();
                    if !data.is_empty() {
                        data.remove(0);
                    }
                    data.insert(0, collecting_data);
                    drop(data);

                    // Log collecting table
                    self.update_solo_collecting_table(verbose);
                    println!("{}: {}", "ERROR".bold().red(), err);

                    return;
                }
            }
        }
    }

    async fn collect_pool(&self, args: CollectArgs, pool: &Pool) -> Result<(), Error> {
        // Register, if needed
        let pool_member = pool.post_pool_register(self).await?;
        let nonce_index = pool_member.id as u64;

        // Get device id
        let device_id = args.device_id.unwrap_or(0);

        // Get verbose flag
        let verbose = args.verbose;

        // Check num threads
        let cores = self.parse_cores(args.cores);
        self.check_num_cores(cores);

        // Init channel for continuous submission
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Solution>();
        tokio::spawn({
            let miner = self.clone();
            let pool = pool.clone();
            async move {
                while let Some(solution) = rx.recv().await {
                    if let Err(err) = pool.post_pool_solution(&miner, &solution).await {
                        println!("error submitting solution: {:?}", err);
                    }
                }
            }
        });

        // Start collecting loop
        let mut last_hash_at = 0;
        loop {
            // Fetch latest challenge
            let member_challenge = match pool.get_updated_pool_challenge(self, last_hash_at).await {
                Err(_err) => {
                    tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                    continue;
                }
                Ok(member_challenge) => member_challenge,
            };

            // Log collecting table
            self.update_pool_collecting_table(verbose);

            // Increment last balance and hash
            last_hash_at = member_challenge.challenge.lash_hash_at;

            // Fetch clock once per loop iteration
            let clock = get_clock(&self.rpc_client)
                .await
                .expect("Failed to fetch clock account");

            // Compute cutoff time
            let cutoff_time = self.get_cutoff(&clock, last_hash_at, args.buffer_time);

            // Build nonce indices
            let num_total_members = member_challenge.num_total_members.max(1);
            let member_search_space_size = u64::MAX.saturating_div(num_total_members);
            let device_search_space_size =
                member_search_space_size.saturating_div(member_challenge.num_devices as u64);

            // Check device id doesn't go beyond pool limit
            if (device_id as u8) > member_challenge.num_devices {
                return Err(Error::TooManyDevices);
            }

            // Calculate bounds on nonce space
            let _left_bound = member_search_space_size.saturating_mul(nonce_index)
                + device_id.saturating_mul(device_search_space_size);

            // Run drillx
            let solution = Self::find_hash_par(
                member_challenge.challenge.challenge,
                cutoff_time,
                cores,
                member_challenge.challenge.min_difficulty as u32,
                Some(tx.clone()),
            )
            .await;

            // Post solution to pool server
            match pool.post_pool_solution(self, &solution).await {
                Err(_err) => {
                    tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                    continue;
                }
                Ok(()) => {
                    self.fetch_pool_collect_event(pool, last_hash_at, verbose)
                        .await;
                }
            }
        }
    }

    async fn find_hash_par(
        challenge: [u8; 32],
        cutoff_time: u64,
        cores: u64,
        min_difficulty: u32,
        pool_channel: Option<tokio::sync::mpsc::UnboundedSender<Solution>>,
    ) -> Solution {
        #[cfg(feature = "gpu")]
        {
            // 判断是否通过命令行参数启用了GPU
            if std::env::var("BITZ_USE_GPU").unwrap_or_default() == "1" {
                return Self::find_hash_gpu(challenge, cutoff_time, min_difficulty, pool_channel).await;
            }
        }

        // CPU实现保持不变
        // Dispatch job to each thread
        let progress_bar = Arc::new(spinner::new_progress_bar());
        let global_best_difficulty = Arc::new(AtomicU32::new(0u32));

        progress_bar.set_message("Collecting...");
        let core_ids = core_affinity::get_core_ids().expect("Failed to fetch core count");
        let core_ids = core_ids.into_iter().filter(|id| id.id < (cores as usize));
        let handles: Vec<_> = core_ids
            .map(|i| {
                let global_best_difficulty = Arc::clone(&global_best_difficulty);
                std::thread::spawn({
                    let progress_bar = progress_bar.clone();
                    let mut memory = equix::SolverMemory::new();
                    let pool_channel = pool_channel.clone();
                    move || {
                        // Pin to core
                        let _ = core_affinity::set_for_current(i);

                        // Start hashing
                        let timer = Instant::now();
                        let mut nonce = rand::thread_rng().gen::<u64>();
                        let mut best_nonce = nonce;
                        let mut best_difficulty = 0;
                        let mut best_hash = Hash::default();
                        loop {
                            // Get hashes
                            let hxs = drillx::hashes_with_memory(
                                &mut memory,
                                &challenge,
                                &nonce.to_le_bytes(),
                            );

                            // Look for best difficulty score in all hashes
                            for hx in hxs {
                                let difficulty = hx.difficulty();
                                if difficulty.gt(&best_difficulty) {
                                    best_nonce = nonce;
                                    best_difficulty = difficulty;
                                    best_hash = hx;
                                    let current_global_best = global_best_difficulty.load(Ordering::Relaxed);
                                    if best_difficulty.gt(&current_global_best)
                                    {
                                        global_best_difficulty.fetch_max(best_difficulty, Ordering::Relaxed);

                                        // Continuously upload best solution to pool
                                        if difficulty.ge(&min_difficulty) {
                                            if let Some(ref ch) = pool_channel {
                                                let digest = best_hash.d;
                                                let nonce = nonce.to_le_bytes();
                                                let solution = Solution {
                                                    d: digest,
                                                    n: nonce,
                                                };
                                                if let Err(err) = ch.send(solution) {
                                                    println!("{} {:?}", "ERROR".bold().red(), err);
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            // Exit if time has elapsed
                            if nonce % 100 == 0 {
                                let current_global_best = global_best_difficulty.load(Ordering::Relaxed);
                                if timer.elapsed().as_secs().ge(&cutoff_time) {
                                    if i.id == 0 {
                                        progress_bar.set_message(format!(
                                            "Collecting...\n  Best score: {}",
                                            current_global_best,
                                        ));
                                    }
                                    if current_global_best.ge(&min_difficulty) {
                                        // Collect until min difficulty has been met
                                        break;
                                    }
                                } else if i.id == 0 {
                                    progress_bar.set_message(format!(
                                        "Collecting...\n  Best score: {}\n  Time remaining: {}",
                                        current_global_best,
                                        format_duration(
                                            cutoff_time.saturating_sub(timer.elapsed().as_secs())
                                                as u32
                                        ),
                                    ));
                                }
                            }

                            // Increment nonce
                            nonce += 1;
                        }

                        // Return the best nonce
                        (best_nonce, best_difficulty, best_hash)
                    }
                })
            })
            .collect();

        // Join handles and return best nonce
        let mut best_nonce: u64 = 0;
        let mut best_difficulty = 0;
        let mut best_hash = Hash::default();
        for h in handles {
            if let Ok((nonce, difficulty, hash)) = h.join() {
                if difficulty > best_difficulty {
                    best_difficulty = difficulty;
                    best_nonce = nonce;
                    best_hash = hash;
                }
            }
        }

        Solution::new(best_hash.d, best_nonce.to_le_bytes())
    }

    #[cfg(feature = "gpu")]
    async fn find_hash_gpu(
        challenge: [u8; 32],
        cutoff_time: u64,
        min_difficulty: u32,
        pool_channel: Option<tokio::sync::mpsc::UnboundedSender<Solution>>,
    ) -> Solution {
        // 获取环境变量中设置的GPU参数
        let batch_size = std::env::var("BITZ_GPU_BATCH_SIZE")
            .map(|v| v.parse::<u32>().unwrap_or(2048))  // 提高默认批处理大小
            .unwrap_or(2048);
        
        let hash_threads = std::env::var("BITZ_GPU_HASH_THREADS")
            .map(|v| v.parse::<u32>().unwrap_or(1024))  // 提高默认线程数
            .unwrap_or(1024);
            
        let solve_threads = std::env::var("BITZ_GPU_SOLVE_THREADS")
            .map(|v| v.parse::<u32>().unwrap_or(256))  // 提高默认解算线程数
            .unwrap_or(256);
        
        let threads = num_cpus::get();
        let progress_bar = Arc::new(spinner::new_progress_bar());
        progress_bar.set_message(format!(
            "Collecting with GPU... (Batch: {}, Hash Threads: {}, Solve Threads: {})",
            batch_size, hash_threads, solve_threads
        ));
    
        let timer = Instant::now();
    
        const INDEX_SPACE: usize = 65536;
        let x_batch_size = batch_size;
        let mut x_nonce = rand::thread_rng().gen::<u64>();
        let mut processed = 0;
    
        // 共享状态
        let xbest = Arc::new(std::sync::Mutex::new((0, 0, Hash::default())));
    
        loop {
            // 分配内存用于结果
            let mut digest = vec![0u8; x_batch_size as usize * 16]; // 每个解决方案16字节
            let mut sols = vec![0u32; x_batch_size as usize];
            
            unsafe {
                // 使用新的统一GPU挖矿函数，避免中间数据交换
                // 设置环境变量
                std::env::set_var("BITZ_GPU_HASH_THREADS", hash_threads.to_string());
                std::env::set_var("BITZ_GPU_SOLVE_THREADS", solve_threads.to_string());
                std::env::set_var("BITZ_GPU_BATCH_SIZE", batch_size.to_string());
                
                // 直接调用统一函数
                gpu_unified_mine(
                    challenge.as_ptr(),
                    &x_nonce as *const u64 as *const u8,
                    digest.as_mut_ptr(),
                    sols.as_mut_ptr(),
                    x_batch_size as i32,
                );
            }
            
            // 使用Rayon并行处理结果
            let chunk_size = x_batch_size as usize / threads;
            let handles: Vec<(u64, u32, Hash)> = (0..threads).into_par_iter().map(|i| {
                let start = i * chunk_size;
                let end = if i + 1 == threads { x_batch_size as usize } else { start + chunk_size };
    
                let mut best_nonce = 0;
                let mut best_difficulty = 0;
                let mut best_hash = Hash::default();
    
                for i in start..end {
                    if sols[i] > 0 {
                        let solution_digest = &digest[i * 16..(i + 1) * 16];
                        let solution = Solution::new(
                            solution_digest.try_into().unwrap(), 
                            (x_nonce + i as u64).to_le_bytes()
                        );
                        let difficulty = solution.to_hash().difficulty();
                        if solution.is_valid(&challenge) && difficulty > best_difficulty {
                            best_nonce = u64::from_le_bytes(solution.n);
                            best_difficulty = difficulty;
                            best_hash = solution.to_hash();
                        }
                    }
                }
    
                (best_nonce, best_difficulty, best_hash)
            }).collect();
    
            // 更新共享状态
            {
                let mut xbest = xbest.lock().unwrap();
                let best_result = handles.into_iter().max_by_key(|&(_, diff, _)| diff).unwrap_or((0, 0, Hash::default()));
                
                // 持续上传最佳解决方案到池
                if best_result.1 >= min_difficulty && best_result.1 > xbest.1 {
                    if let Some(ref ch) = pool_channel {
                        let digest = best_result.2.d;
                        let nonce = best_result.0.to_le_bytes();
                        let solution = Solution {
                            d: digest,
                            n: nonce,
                        };
                        if let Err(err) = ch.send(solution) {
                            println!("{} {:?}", "ERROR".bold().red(), err);
                        }
                    }
                }
                
                // 更新最佳结果
                if best_result.1 > xbest.1 {
                    *xbest = best_result;
                }
            }
    
            // 为下一批增加nonce
            x_nonce += x_batch_size as u64;
            processed += x_batch_size as usize;
    
            // 更新进度条
            let elapsed = timer.elapsed().as_secs();
            let best_difficulty = {
                let xbest = xbest.lock().unwrap();
                xbest.1
            };
    
            progress_bar.set_message(format!(
                "Collecting with GPU... (Best score: {}, Time Remaining: {}s, Processed: {})",
                best_difficulty,
                cutoff_time.saturating_sub(elapsed),
                processed
            ));
    
            if timer.elapsed().as_secs() >= cutoff_time {
                let xbest = xbest.lock().unwrap();
                if xbest.1 >= min_difficulty {
                    break;
                }
            }
        }
    
        let final_best = xbest.lock().unwrap(); // 锁定并获取最终最佳结果
        progress_bar.finish_with_message(format!(
            "Best hash: {} (score: {})",
            bs58::encode(final_best.2.h).into_string(),
            final_best.1
        ));
    
        Solution::new(final_best.2.d, final_best.0.to_le_bytes())
    }

    pub fn parse_cores(&self, cores: String) -> u64 {
        if cores == "ALL" {
            num_cpus::get() as u64
        } else {
            cores.parse::<u64>().unwrap()
        }
    }

    pub fn check_num_cores(&self, cores: u64) {
        let num_cores = num_cpus::get() as u64;
        if cores.gt(&num_cores) {
            println!(
                "{} Cannot exceeds available cores ({})",
                "WARNING".bold().yellow(),
                num_cores
            );
        }
    }

    async fn should_reset(&self, clock: &solana_program::clock::Clock, config: Config) -> bool {
        config
            .last_reset_at
            .saturating_add(EPOCH_DURATION)
            .saturating_sub(5) // Buffer
            .le(&clock.unix_timestamp)
    }

    fn get_cutoff(&self, clock: &solana_program::clock::Clock, last_hash_at: i64, buffer_time: u64) -> u64 {
        last_hash_at
            .saturating_add(60)
            .saturating_sub(buffer_time as i64)
            .saturating_sub(clock.unix_timestamp)
            .max(0) as u64
    }

    async fn find_bus(&self) -> Pubkey {
        // Fetch the bus with the largest balance
        if let Ok(accounts) = self.rpc_client.get_multiple_accounts(&BUS_ADDRESSES).await {
            let mut top_bus_balance: u64 = 0;
            let mut top_bus = BUS_ADDRESSES[0];
            for account in accounts {
                if let Some(account) = account {
                    if let Ok(bus) = Bus::try_from_bytes(&account.data) {
                        if bus.rewards.gt(&top_bus_balance) {
                            top_bus_balance = bus.rewards;
                            top_bus = BUS_ADDRESSES[bus.id as usize];
                        }
                    }
                }
            }
            return top_bus;
        }

        // Otherwise return a random bus
        let i = rand::thread_rng().gen_range(0..BUS_COUNT);
        BUS_ADDRESSES[i]
    }

    async fn fetch_solo_collect_event(&self, sig: Signature, verbose: bool) {
        // Add loading row
        let collecting_data = SoloCollectingData::fetching(sig);
        let mut data = self.solo_collecting_data.write().unwrap();
        data.insert(0, collecting_data);
        if !data.is_empty() {
            data.remove(0);
        }
        drop(data);

        // Update table
        self.update_solo_collecting_table(verbose);

        // Poll for transaction
        let mut tx;
        let mut attempts = 0;
        loop {
            tx = self
                .rpc_client
                .get_transaction(&sig, UiTransactionEncoding::Json)
                .await;
            if tx.is_ok() {
                break;
            }
            sleep(Duration::from_secs(1));
            attempts += 1;
            if attempts > 30 {
                break;
            }
        }

        // Parse transaction response
        if let Ok(tx) = tx {
            if let Some(meta) = tx.transaction.meta {
                if let OptionSerializer::Some(log_messages) = meta.log_messages {
                    if let Some(return_log) = log_messages
                        .iter()
                        .find(|log| log.starts_with("Program return: "))
                    {
                        if let Some(return_data) =
                            return_log.strip_prefix(&format!("Program return: {} ", eore_api::ID))
                        {
                            if let Ok(return_data) = return_data.from_base64() {
                                let mut data = self.solo_collecting_data.write().unwrap();
                                let event = MineEvent::from_bytes(&return_data);
                                let collecting_data = SoloCollectingData {
                                    signature: if verbose {
                                        sig.to_string()
                                    } else {
                                        format!("{}...", sig.to_string()[..8].to_string())
                                    },
                                    block: tx.slot.to_string(),
                                    timestamp: format_timestamp(tx.block_time.unwrap_or_default()),
                                    difficulty: event.difficulty.to_string(),
                                    base_reward: if event.net_base_reward > 0 {
                                        format!("{:#.11}", amount_u64_to_f64(event.net_base_reward))
                                    } else {
                                        "0".to_string()
                                    },
                                    boost_reward: if event.net_miner_boost_reward > 0 {
                                        format!(
                                            "{:#.11}",
                                            amount_u64_to_f64(event.net_miner_boost_reward)
                                        )
                                    } else {
                                        "0".to_string()
                                    },
                                    total_reward: if event.net_reward > 0 {
                                        format!("{:#.11}", amount_u64_to_f64(event.net_reward))
                                    } else {
                                        "0".to_string()
                                    },
                                    timing: format!("{}s", event.timing),
                                    status: "Confirmed".bold().green().to_string(),
                                };
                                data.insert(0, collecting_data);
                            }
                        }
                    }
                }
            }
        }
    }

    async fn fetch_pool_collect_event(&self, pool: &Pool, last_hash_at: i64, verbose: bool) {
        let collecting_data = match pool
            .get_latest_pool_event(self.signer().pubkey(), last_hash_at)
            .await
        {
            Ok(event) => PoolCollectingData {
                signature: if verbose {
                    event.signature.to_string()
                } else {
                    format!("{}...", event.signature.to_string()[..8].to_string())
                },
                block: event.block.to_string(),
                timestamp: format_timestamp(event.timestamp as i64),
                timing: format!("{}s", event.timing),
                difficulty: event.difficulty.to_string(),
                base_reward: if event.net_base_reward > 0 {
                    format!("{:#.11}", amount_u64_to_f64(event.net_base_reward))
                } else {
                    "0".to_string()
                },
                boost_reward: if event.net_miner_boost_reward > 0 {
                    format!("{:#.11}", amount_u64_to_f64(event.net_miner_boost_reward))
                } else {
                    "0".to_string()
                },
                total_reward: if event.net_reward > 0 {
                    format!("{:#.11}", amount_u64_to_f64(event.net_reward))
                } else {
                    "0".to_string()
                },
                my_difficulty: event.member_difficulty.to_string(),
                my_reward: if event.member_reward > 0 {
                    format!("{:#.11}", amount_u64_to_f64(event.member_reward))
                } else {
                    "0".to_string()
                },
            },
            Err(err) => PoolCollectingData {
                signature: format!("Failed to fetch event: {:?}", err),
                block: "".to_string(),
                timestamp: "".to_string(),
                timing: "".to_string(),
                difficulty: "".to_string(),
                base_reward: "".to_string(),
                boost_reward: "".to_string(),
                total_reward: "".to_string(),
                my_difficulty: "".to_string(),
                my_reward: "".to_string(),
            },
        };

        // Add row
        let mut data = self.pool_collecting_data.write().unwrap();
        data.insert(0, collecting_data);
        if data.len() >= 12 {
            data.pop();
        }
        drop(data);
    }

    fn update_solo_collecting_table(&self, verbose: bool) {
        execute!(stdout(), Clear(ClearType::All), MoveTo(0, 0)).unwrap();
        let mut rows: Vec<SoloCollectingData> = vec![];
        let data = self.solo_collecting_data.read().unwrap();
        rows.extend(data.iter().cloned());
        let mut table = Table::new(&rows);
        table.with(Style::blank());
        table.modify(Columns::new(1..), Alignment::right());
        table.modify(Rows::first(), Color::BOLD);
        table.with(
            Highlight::new(Rows::single(1)).color(BorderColor::default().top(Color::FG_WHITE)),
        );
        table.with(Highlight::new(Rows::single(1)).border(Border::new().top('━')));
        if !verbose {
            table.with(Remove::column(Columns::new(1..3)));
        }
        println!("\n{}\n", table);
    }

    fn update_pool_collecting_table(&self, verbose: bool) {
        execute!(stdout(), Clear(ClearType::All), MoveTo(0, 0)).unwrap();
        let mut rows: Vec<PoolCollectingData> = vec![];
        let data = self.pool_collecting_data.read().unwrap();
        rows.extend(data.iter().cloned());
        let mut table = Table::new(&rows);
        table.with(Style::blank());
        table.modify(Columns::new(1..), Alignment::right());
        table.modify(Rows::first(), Color::BOLD);
        table.with(
            Highlight::new(Rows::single(1)).color(BorderColor::default().top(Color::FG_WHITE)),
        );
        table.with(Highlight::new(Rows::single(1)).border(Border::new().top('━')));
        if !verbose {
            table.with(Remove::column(Columns::new(1..3)));
        }
        println!("\n{}\n", table);
    }

    async fn open(&self) {
        // Register collector
        let mut ixs = Vec::new();
        let signer = self.signer();
        let fee_payer = self.fee_payer();
        let proof_address = proof_pda(signer.pubkey()).0;
        if self.rpc_client.get_account(&proof_address).await.is_err() {
            let ix = eore_api::sdk::open(signer.pubkey(), signer.pubkey(), fee_payer.pubkey());
            ixs.push(ix);
        }

        // Submit transaction
        if ixs.len() > 0 {
            self.send_and_confirm(&ixs, ComputeBudget::Fixed(400_000), false)
                .await
                .ok();
        }
    }
} 
