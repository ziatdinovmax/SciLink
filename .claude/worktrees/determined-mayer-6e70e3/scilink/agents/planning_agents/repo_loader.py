import subprocess
import os
from pathlib import Path
from urllib.parse import urlparse

def clone_git_repository(repo_url: str, 
                         target_base_dir: str = "./downloaded_repos", 
                         auto_update: bool = True) -> str:
    """
    Clones a git repository to a local directory.
    If the directory exists and auto_update is True, it runs 'git pull'.
    
    Returns the absolute path to the cloned directory.
    """
    # 1. Extract repo name to use as folder name
    # e.g., https://github.com/user/my-project.git -> my-project
    parsed_url = urlparse(repo_url)
    repo_name = os.path.basename(parsed_url.path)
    if repo_name.endswith('.git'):
        repo_name = repo_name[:-4]
    
    # Clean up name to ensure valid folder path
    repo_name = "".join(c for c in repo_name if c.isalnum() or c in ('-', '_'))
    
    target_path = Path(target_base_dir) / repo_name
    
    # 2. Check if git is installed
    try:
        subprocess.run(["git", "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("  - ❌ Error: 'git' is not installed or not in PATH.")
        return None

    # 3. Handle Existing Directory
    if target_path.exists():
        if auto_update:
            print(f"  - 🔄 Repo '{repo_name}' exists. Attempting update (git pull)...")
            try:
                # 'git -C path' runs the command inside that directory
                subprocess.run(["git", "-C", str(target_path), "pull"], 
                               check=True, 
                               stdout=subprocess.DEVNULL) # Hide generic output unless error
                print(f"  - ✅ Update successful: {repo_name}")
            except subprocess.CalledProcessError as e:
                print(f"  - ⚠️  Update failed (local changes or network issue): {e}")
                print("       Using existing version without update.")
        else:
            print(f"  - ℹ️  Repo '{repo_name}' exists. Skipping update.")
            
        return str(target_path.resolve())
    
    # 4. Clone New Repo
    print(f"  - 📥 Cloning '{repo_url}' into {target_path}...")
    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", repo_url, str(target_path)], check=True)
        print("  - ✅ Clone successful.")
        return str(target_path.resolve())
    except subprocess.CalledProcessError as e:
        print(f"  - ❌ Error cloning repo: {e}")
        return None