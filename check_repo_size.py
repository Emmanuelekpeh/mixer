#!/usr/bin/env python3
"""
🔍 Git Repository Size Check
==========================

This script checks the size of files that would be included in a Git commit
based on the current .gitignore settings. It helps identify any large files
that might accidentally be committed.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_repo_size():
    """Check the size of files that would be included in a Git commit"""
    print("🔍 Checking Repository Size")
    print("=" * 50)
    
    # Get list of files that would be committed (not ignored by .gitignore)
    try:
        # This command lists all files that Git would track (not ignored)
        result = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            capture_output=True, text=True, check=True
        )
        untracked_files = result.stdout.splitlines()
        
        result = subprocess.run(
            ["git", "ls-files"],
            capture_output=True, text=True, check=True
        )
        tracked_files = result.stdout.splitlines()
        
        all_files = tracked_files + untracked_files
        
        if not all_files:
            print("No files found that would be included in a Git commit.")
            return
        
        # Collect file sizes
        file_sizes = []
        total_size = 0
        
        for file_path in all_files:
            path = Path(file_path)
            if path.exists():
                size = path.stat().st_size
                size_mb = size / (1024 * 1024)
                total_size += size
                
                file_sizes.append({
                    "path": file_path,
                    "size_bytes": size,
                    "size_mb": size_mb
                })
        
        # Sort by size (largest first)
        file_sizes.sort(key=lambda x: x["size_bytes"], reverse=True)
        
        # Print largest files
        print(f"\nLargest files that would be included in a Git commit:")
        print("-" * 80)
        print(f"{'Size (MB)':>10} | Path")
        print("-" * 80)
        
        large_file_threshold = 1  # MB
        large_files = []
        
        for file_info in file_sizes[:20]:  # Show top 20 largest files
            print(f"{file_info['size_mb']:10.2f} | {file_info['path']}")
            
            if file_info['size_mb'] > large_file_threshold:
                large_files.append(file_info)
        
        # Print summary
        print("\n📊 Summary:")
        print(f"Total files: {len(file_sizes)}")
        print(f"Total size: {total_size / (1024 * 1024):.2f} MB")
        
        if large_files:
            print(f"\n⚠️ Large files (>{large_file_threshold} MB) found:")
            for file_info in large_files:
                print(f"  - {file_info['path']} ({file_info['size_mb']:.2f} MB)")
            
            print("\nConsider adding these files to .gitignore if they shouldn't be in the repository.")
        else:
            print("\n✅ No unusually large files found.")
        
    except subprocess.CalledProcessError as e:
        print(f"Error running Git command: {e}")
        return

if __name__ == "__main__":
    check_repo_size()
