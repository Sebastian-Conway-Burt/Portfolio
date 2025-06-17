3. Collaboration Workflow with Your Teammate

Add Teammate as Collaborator:

On your GitHub repository page, go to Settings -> Collaborators and teams.
Click Add people and invite your teammate using their GitHub username or email. They will need to accept the invitation.
Teammate Clones the Repository: Your teammate needs to get a copy of the project onto their machine (or their own JupyterHub environment). They run this command once in the parent directory where they want the project folder to live:

Bash

# Teammate runs this, replacing the URL
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
This creates the EDU-Copilot folder on their end, already linked to the remote repository.

Standard Workflow (Both of you follow this):

Pull Changes: Before starting any work, always get the latest changes from GitHub to avoid conflicts:

Bash

git pull origin main
(Or just git pull if the upstream is set correctly).

Make Changes: Edit HTML, CSS, or JS files locally in VS Code as needed.

Stage Changes: Add the specific files you modified or use git add . for all changes.

Bash

git add path/to/your/changed_file.html css/style.css
# OR
git add .
Commit Changes: Save your work with a descriptive message.

Bash

git commit -m "Feat: Add styling for scholarship page"
# or -m "Fix: Corrected link in navbar"
Push Changes: Upload your committed changes to GitHub for your teammate to see/pull.

Bash

git push origin main
(Or just git push).

Handling Conflicts: If both of you edit the same lines in the same file before pulling, Git might detect a "merge conflict" when you git pull. VS Code has tools to help resolve these. The key is frequent communication and pulling changes often!


Need to add images to image section
