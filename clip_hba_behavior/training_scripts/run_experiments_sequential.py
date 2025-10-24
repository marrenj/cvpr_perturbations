#!/usr/bin/env python3
"""
Sequential experiment runner for perturbation sweep.
Reads experiment_runs.tsv and runs each training experiment one by one.
"""

import pandas as pd
import subprocess
import sys
import os
from datetime import datetime
import time
import threading

def run_experiment(command, experiment_id, log_dir):
    """
    Run a single training experiment with real-time console output.
    """
    # Create log files for this experiment
    log_file = os.path.join(log_dir, f"{experiment_id}.out")
    error_file = os.path.join(log_dir, f"{experiment_id}.err")
    
    print(f"\n{'='*80}")
    print(f"STARTING EXPERIMENT {experiment_id}")
    print(f"Command: {command}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log file: {log_file}")
    print(f"Error file: {error_file}")
    print(f"{'='*80}")
    
    try:
        # Run the command with real-time output to console + logging to files
        with open(log_file, 'w') as out_f, open(error_file, 'w') as err_f:
            # Set environment variables for unbuffered output
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            
            process = subprocess.Popen(
                command.split(),
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0,  # Unbuffered for real-time output
                universal_newlines=True,
                env=env
            )

            # Stream output in real-time using threads
            
            def stream_output(pipe, file_handle, is_stderr=False):
                for line in iter(pipe.readline, ''):
                    if line:
                        if is_stderr:
                            print(line.rstrip(), file=sys.stderr)  # Show errors in console
                        else:
                            print(line.rstrip())  # Show in console
                        file_handle.write(line)    # Save to log file
                        file_handle.flush()
                pipe.close()
            
            # Start threads for stdout and stderr
            stdout_thread = threading.Thread(target=stream_output, args=(process.stdout, out_f))
            stderr_thread = threading.Thread(target=stream_output, args=(process.stderr, err_f, True))
            
            stdout_thread.start()
            stderr_thread.start()
            
            # Wait for process to complete
            return_code = process.wait()
            
            # Wait for threads to finish
            stdout_thread.join()
            stderr_thread.join()
            
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, command)
        
        print(f"\n✅ EXPERIMENT {experiment_id} COMPLETED SUCCESSFULLY")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n❌ EXPERIMENT {experiment_id} FAILED")
        print(f"Error code: {e.returncode}")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return False
    except Exception as e:
        print(f"\n❌ EXPERIMENT {experiment_id} FAILED WITH EXCEPTION")
        print(f"Error: {str(e)}")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return False

def main():
    """Main function to run all experiments sequentially."""
    
    # Create timestamped logs directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"/home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/clip_hba_behavior_loops/logs/{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    print(f"📁 Logs directory: {log_dir}")
    print(f"🕒 Timestamp: {timestamp}")
    
    # Read the experiment runs TSV file
    tsv_file = "experiment_runs.tsv"
    
    if not os.path.exists(tsv_file):
        print(f"❌ Error: {tsv_file} not found!")
        print("Make sure you're running this script from the training_scripts directory.")
        sys.exit(1)
    
    # Load experiments
    try:
        df = pd.read_csv(tsv_file, sep='\t')
        print(f"📋 Loaded {len(df)} experiments from {tsv_file}")
    except Exception as e:
        print(f"❌ Error reading {tsv_file}: {e}")
        sys.exit(1)
    
    # Track results
    successful_runs = []
    failed_runs = []
    start_time = datetime.now()
    
    print(f"\n🚀 STARTING SEQUENTIAL EXPERIMENT RUN")
    print(f"Total experiments: {len(df)}")
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    # Run each experiment
    for idx, row in df.iterrows():
        experiment_id = row['run_id']
        command = row['command']
        
        # Add -u flag to force unbuffered output
        if command.startswith('python '):
            command = command.replace('python ', 'python -u ', 1)
        
        # Run the experiment
        success = run_experiment(command, experiment_id, log_dir)
        
        if success:
            successful_runs.append(experiment_id)
        else:
            failed_runs.append(experiment_id)
        
        # Print progress
        completed = len(successful_runs) + len(failed_runs)
        remaining = len(df) - completed
        print(f"\n📊 PROGRESS: {completed}/{len(df)} experiments completed")
        print(f"✅ Successful: {len(successful_runs)}")
        print(f"❌ Failed: {len(failed_runs)}")
        print(f"⏳ Remaining: {remaining}")
        
        # Ask user if they want to continue after failures
        if failed_runs and len(failed_runs) == 1:  # First failure
            print(f"\n⚠️  First failure detected. Do you want to continue?")
            response = input("Continue with remaining experiments? (y/n): ").lower().strip()
            if response != 'y':
                print("🛑 Stopping experiment run.")
                break
    
    # Final summary
    end_time = datetime.now()
    total_time = end_time - start_time
    
    print(f"\n{'='*80}")
    print(f"🏁 EXPERIMENT RUN COMPLETED")
    print(f"{'='*80}")
    print(f"Total time: {total_time}")
    print(f"Total experiments: {len(df)}")
    print(f"✅ Successful: {len(successful_runs)}")
    print(f"❌ Failed: {len(failed_runs)}")
    
    if successful_runs:
        print(f"\n✅ Successful experiments:")
        for run_id in successful_runs:
            print(f"  - {run_id}")
    
    if failed_runs:
        print(f"\n❌ Failed experiments:")
        for run_id in failed_runs:
            print(f"  - {run_id}")
    
    print(f"\n📁 Results saved to:")
    print(f"  - Experiment logs: {log_dir}/")
    print(f"  - Training results: /home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/clip_hba_behavior_loops/{timestamp}/output/")

if __name__ == "__main__":
    main()
