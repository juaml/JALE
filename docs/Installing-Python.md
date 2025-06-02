This guide walks you through the steps to set up Python and manage virtual environments using `pyenv` and `pyenv-virtualenv` on **macOS** and **Linux**.

---

## Prerequisites

Before you start, ensure you have the following installed:

### macOS

- **Homebrew**:
  
  Check if Homebrew is already installed:
  ```bash
  brew
  ```
 
  If this throws an error: 
  ```bash
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  ```
  Follow the instructions provided by the Homebrew installer to add `brew` to your path.


- Install required dependencies:
  ```bash
  brew install openssl readline sqlite3 xz zlib tcl-tk git libomp
  ```

### Linux (Debian/Ubuntu-based systems)

- Install build dependencies:
  ```bash
  sudo apt update && sudo apt install -y \
      make build-essential libssl-dev zlib1g-dev \
      libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
      libncurses5-dev libncursesw5-dev xz-utils tk-dev \
      libffi-dev liblzma-dev python-openssl git
  ```

---

## Step 1: Install `pyenv`

1. Run the following command to install `pyenv`:

   ```bash
   curl https://pyenv.run | bash
   ```

   Alternatively, use `wget`:

   ```bash
   wget -qO- https://pyenv.run | bash
   ```

2. Find out which configuration file your shell is loading. Most common: `~/.bashrc`, `~/.zshrc` or `~/.bash_profile`.

   e.g.
   ```bash
   echo $SHELL
   ```


2. Run the command corresponding to your shell configuration file. (`~/.bashrc`, `~/.zshrc`, etc.):

   ```bash
   echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
   echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
   echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
   echo 'eval "$(pyenv init -)"' >> ~/.bashrc
   ```

   ```bash
   echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
   echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
   echo 'eval "$(pyenv init --path)"' >> ~/.zshrc
   echo 'eval "$(pyenv init -)"' >> ~/.zshrc
   ```

3. Reload your shell:

   ```bash
   exec $SHELL
   ```

4. Verify the installation:

   ```bash
   pyenv --version
   ```

---

## Step 2: Install `pyenv-virtualenv`

1. Install the `pyenv-virtualenv` plugin:

   ```bash
   git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv
   ```

2. Add the following line to your shell configuration file (`~/.bashrc`, `~/.zshrc`, etc.):

   ```bash
   eval "$(pyenv virtualenv-init -)"
   ```

3. Reload your shell:

   ```bash
   exec $SHELL
   ```

4. Verify the installation:

   ```bash
   pyenv virtualenv --version
   ```

---

## Step 3: Install a Python Version and setup a virtual environment

1. Install Python Version 3.11

   ```bash
   pyenv install 3.11
   ```

3. Create a virtual environment with a name of your choice:

   ```bash
   pyenv virtualenv 3.11 <env-name>
   ```

4. Set the virtual environment as the global Python Environment:

   ```bash
   pyenv global <env-name>
   ```


### You are now ready to install JALE. [Installing JALE](Installing-JALE)

---
---

### Useful pyenv commands

- List all available Python versions:

   ```bash
   pyenv install --list
   ```

- Install a specific Python version:

   ```bash
   pyenv install <version>
   ```

   Example:

   ```bash
   pyenv install 3.11.5
   ```

- Set the global Python version:

   ```bash
   pyenv global <version>
   ```

   Example:

   ```bash
   pyenv global 3.11.5
   ```

- Verify the installed Python version:

   ```bash
   python --version
   ```

- Check all installed Python versions:

   ```bash
   pyenv versions
   ```

- Create a virtual environment:

   ```bash
   pyenv virtualenv <python-version> <env-name>
   ```

   Example:

   ```bash
   pyenv virtualenv 3.11.5 myenv
   ```

- Activate the virtual environment:

   ```bash
   pyenv activate <env-name>
   ```

   Example:

   ```bash
   pyenv activate myenv
   ```

- Deactivate the virtual environment:

   ```bash
   pyenv deactivate
   ```

- List all virtual environments:

   ```bash
   pyenv virtualenvs
   ```

- Set a local Python version or virtual environment for a specific project directory:

   ```bash
   pyenv local <version-or-env>
   ```

   Example:

   ```bash
   pyenv local myenv
   ```

- Remove a virtual environment:

   ```bash
   pyenv uninstall <env-name>
   ```

---

## Troubleshooting

### macOS Specific Issues

- If you use macOS with M1/M2 chips, consider setting additional flags when installing Python:
  ```bash
  CFLAGS="-I$(brew --prefix openssl)/include" \
  LDFLAGS="-L$(brew --prefix openssl)/lib" \
  pyenv install <version>
  ```

### Linux Specific Issues
- If you encounter build issues while installing Python, ensure all required dependencies (listed in prerequisites) are installed.
