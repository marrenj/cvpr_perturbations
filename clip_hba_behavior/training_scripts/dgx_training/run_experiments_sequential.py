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
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            
            process = subprocess.Popen(
                command.split(),
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0,
                universal_newlines=True,
                env=env
            )

            def stream_output(pipe, file_handle, is_stderr=False):
                for line in iter(pipe.readline, ''):
                    if line:
                        if is_stderr:
                            print(line.rstrip(), file=sys.stderr)
                        else:
                            print(line.rstrip())
                        file_handle.write(line)
                        file_handle.flush()
                pipe.close()
            
            stdout_thread = threading.Thread(target=stream_output, args=(process.stdout, out_f))
            stderr_thread = threading.Thread(target=stream_output, args=(process.stderr, err_f, True))
            
            stdout_thread.start()
            stderr_thread.start()
            
            return_code = process.wait()
            
            stdout_thread.join()
            stderr_thread.join()
            
            if return_code == 2:
                print(f"\n⏭️  EXPERIMENT {experiment_id} SKIPPED (exit code 2)")
                print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                return "skipped"
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, command)

        print(f"\n✅ EXPERIMENT {experiment_id} COMPLETED SUCCESSFULLY")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return "success"

    except subprocess.CalledProcessError as e:
        print(f"\n❌ EXPERIMENT {experiment_id} FAILED")
        print(f"Error code: {e.returncode}")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return "failed"
    except Exception as e:
        print(f"\n❌ EXPERIMENT {experiment_id} FAILED WITH EXCEPTION")
        print(f"Error: {str(e)}")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return "failed"


def main():
    """Main function to run all experiments sequentially."""
    
    # Create timestamped logs directory (✅ updated base path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"/data/p_dsi/dhungs1/clip_hba_behavior/logs/{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    print(f"📁 Logs directory: {log_dir}")
    print(f"🕒 Timestamp: {timestamp}")
    
    # Read the experiment runs TSV file (✅ local to script folder)
    tsv_file = "experiment_runs.tsv"
    
    if not os.path.exists(tsv_file):
        print(f"❌ Error: {tsv_file} not found!")
        print("Make sure you're running this script from the training_scripts directory.")
        sys.exit(1)
    
    try:
        df = pd.read_csv(tsv_file, sep='\t')
        print(f"📋 Loaded {len(df)} experiments from {tsv_file}")
    except Exception as e:
        print(f"❌ Error reading {tsv_file}: {e}")
        sys.exit(1)
    
    successful_runs, failed_runs, skipped_runs = [], [], []
    start_time = datetime.now()
    
    print(f"\n🚀 STARTING SEQUENTIAL EXPERIMENT RUN")
    print(f"Total experiments: {len(df)}")
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    for idx, row in df.iterrows():
        experiment_id = row['run_id']
        command = row['command']
        
        if command.startswith('python '):
            command = command.replace('python ', 'python -u ', 1)
        
        status = run_experiment(command, experiment_id, log_dir)

        if status == "success":
            successful_runs.append(experiment_id)
        elif status == "skipped":
            skipped_runs.append(experiment_id)
        else:
            failed_runs.append(experiment_id)
        
        completed = len(successful_runs) + len(failed_runs) + len(skipped_runs)
        remaining = len(df) - completed
        print(f"\n📊 PROGRESS: {completed}/{len(df)} experiments completed")
        print(f"✅ Successful: {len(successful_runs)}")
        print(f"⏭️  Skipped: {len(skipped_runs)}")
        print(f"❌ Failed: {len(failed_runs)}")
        print(f"⏳ Remaining: {remaining}")
        
        if failed_runs and len(failed_runs) == 1:
            print(f"\n⚠️  First failure detected. Do you want to continue?")
            response = input("Continue with remaining experiments? (y/n): ").lower().strip()
            if response != 'y':
                print("🛑 Stopping experiment run.")
                break
    
    end_time = datetime.now()
    total_time = end_time - start_time
    
    print(f"\n{'='*80}")
    print(f"🏁 EXPERIMENT RUN COMPLETED")
    print(f"{'='*80}")
    print(f"Total time: {total_time}")
    print(f"Total experiments: {len(df)}")
    print(f"✅ Successful: {len(successful_runs)}")
    print(f"⏭️  Skipped: {len(skipped_runs)}")
    print(f"❌ Failed: {len(failed_runs)}")
    
    if successful_runs:
        print(f"\n✅ Successful experiments:")
        for run_id in successful_runs:
            print(f"  - {run_id}")
    
    if skipped_runs:
        print(f"\n⏭️  Skipped experiments:")
        for run_id in skipped_runs:
            print(f"  - {run_id}")

    if failed_runs:
        print(f"\n❌ Failed experiments:")
        for run_id in failed_runs:
            print(f"  - {run_id}")
    
    print(f"\n📁 Results saved to:")
    print(f"  - Experiment logs: {log_dir}/")
    print(f"  - Training results: /data/p_dsi/dhungs1/clip_hba_behavior/{timestamp}/output/")


if __name__ == "__main__":
    main()
