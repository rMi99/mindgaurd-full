#!/usr/bin/env python3

import re

def fix_user_management_indentation():
    with open('app/routes/user_management.py', 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if this is a function definition
        if re.match(r'^async def ', line.strip()):
            fixed_lines.append(line)
            i += 1
            
            # Check if next line has wrong db = await get_db() placement
            if i < len(lines) and '        db = await get_db()' in lines[i]:
                # Skip this line and look for docstring
                db_line = lines[i]
                i += 1
                
                # Find and add docstring
                if i < len(lines) and '"""' in lines[i]:
                    fixed_lines.append(lines[i])  # Add docstring
                    i += 1
                    
                    # Add properly indented db line and try
                    fixed_lines.append('    db = await get_db()\n')
                    if i < len(lines) and lines[i].strip() == 'try:':
                        fixed_lines.append('    try:\n')
                        i += 1
                    else:
                        fixed_lines.append('    try:\n')
                else:
                    # No docstring, just add db line
                    fixed_lines.append('    db = await get_db()\n')
                    fixed_lines.append('    try:\n')
            else:
                # Normal case
                continue
        else:
            fixed_lines.append(line)
            i += 1
    
    with open('app/routes/user_management.py', 'w') as f:
        f.writelines(fixed_lines)

if __name__ == "__main__":
    fix_user_management_indentation()
    print("Fixed indentation issues in user_management.py")
