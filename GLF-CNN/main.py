import subprocess

def run_script(script_name):
    result = subprocess.run(['python', script_name], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {script_name}:\n{result.stderr}")
    else:
        print(f"Successfully ran {script_name}:\n{result.stdout}")

def main():
    scripts = [
        'data_preprocess.py',
        'local_feature_extractor.py',
        'global_feature_extractor.py',
        'feature_fusion.py'
    ]

    for script in scripts:
        run_script(script)

if __name__ == "__main__":
    main()
