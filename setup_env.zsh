#!/usr/bin/env zsh

# Script to set up the environment for the Titans implementation (ZSH version)
# This script takes advantage of zsh-specific features for better user experience

# Exit on error
setopt ERR_EXIT

# Set up colors for better readability
autoload -U colors && colors

# Print usage information
print_usage() {
    echo "${fg[yellow]}Usage: $0 [OPTIONS]${reset_color}"
    echo "Options:"
    echo "  ${fg[green]}--conda${reset_color}       Use conda environment (default)"
    echo "  ${fg[green]}--venv${reset_color}        Use Python virtual environment"
    echo "  ${fg[green]}--cpu${reset_color}         Install CPU-only version of PyTorch"
    echo "  ${fg[green]}--cuda${reset_color}        Install CUDA version of PyTorch (default)"
    echo "  ${fg[green]}--help${reset_color}        Display this help message"
}

# Parse command line arguments
USE_CONDA=true
USE_CUDA=true

for arg in "$@"; do
    case $arg in
        --conda)
            USE_CONDA=true
            ;;
        --venv)
            USE_CONDA=false
            ;;
        --cpu)
            USE_CUDA=false
            ;;
        --cuda)
            USE_CUDA=true
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo "${fg[red]}Unknown option: $arg${reset_color}"
            print_usage
            exit 1
            ;;
    esac
done

echo "${fg[blue]}Setting up environment for Titans implementation...${reset_color}"

# Create the environment
if $USE_CONDA; then
    echo "${fg[cyan]}Using conda environment...${reset_color}"
    
    # Check if conda is installed
    if ! command -v conda &> /dev/null; then
        echo "${fg[red]}Conda not found. Please install Anaconda or Miniconda first.${reset_color}"
        exit 1
    fi
    
    # Create conda environment
    if conda env list | grep -q "titans"; then
        echo "${fg[yellow]}Titans conda environment already exists. Updating...${reset_color}"
        # ZSH-specific: Use conda's shell hook for proper activation
        eval "$(conda shell.zsh hook)"
        conda activate titans
    else
        echo "${fg[green]}Creating new conda environment: titans${reset_color}"
        conda create -n titans python=3.9 -y
        # ZSH-specific: Use conda's shell hook for proper activation
        eval "$(conda shell.zsh hook)"
        conda activate titans
    fi
    
    # Install PyTorch
    if $USE_CUDA; then
        echo "${fg[blue]}Installing PyTorch with CUDA support...${reset_color}"
        conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
    else
        echo "${fg[blue]}Installing CPU-only PyTorch...${reset_color}"
        conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
    fi
    
    # Install other dependencies
    echo "${fg[blue]}Installing other dependencies...${reset_color}"
    pip install -r requirements.txt
    
    # Install package in development mode
    echo "${fg[blue]}Installing titans package in development mode...${reset_color}"
    pip install -e .
    
    echo "${fg[green]}Conda environment setup complete.${reset_color}"
    echo "To activate in the future, run: ${fg[yellow]}conda activate titans${reset_color}"
else
    echo "${fg[cyan]}Using Python virtual environment...${reset_color}"
    
    # Check if Python is installed
    if ! command -v python3 &> /dev/null; then
        echo "${fg[red]}Python 3 not found. Please install Python 3 first.${reset_color}"
        exit 1
    fi
    
    # Create virtual environment
    if [[ -d "titans_env" ]]; then
        echo "${fg[yellow]}Virtual environment already exists. Updating...${reset_color}"
    else
        echo "${fg[green]}Creating new virtual environment: titans_env${reset_color}"
        python3 -m venv titans_env
    fi
    
    # ZSH-specific: Activate virtual environment
    source titans_env/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install PyTorch
    if $USE_CUDA; then
        echo "${fg[blue]}Installing PyTorch with CUDA support...${reset_color}"
        pip install torch torchvision torchaudio
    else
        echo "${fg[blue]}Installing CPU-only PyTorch...${reset_color}"
        pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
    fi
    
    # Install other dependencies
    echo "${fg[blue]}Installing other dependencies...${reset_color}"
    pip install -r requirements.txt
    
    # Install package in development mode
    echo "${fg[blue]}Installing titans package in development mode...${reset_color}"
    pip install -e .
    
    echo "${fg[green]}Virtual environment setup complete.${reset_color}"
    echo "To activate in the future, run: ${fg[yellow]}source titans_env/bin/activate${reset_color}"
fi

# Create sample data
echo "${fg[blue]}Creating sample data...${reset_color}"
mkdir -p data/sample
python -m scripts.train --create_sample_data --sample_data_dir data/sample

# Add convenient aliases for zsh
echo "\n${fg[cyan]}Adding convenient aliases to your shell...${reset_color}"
cat <<EOT >> ~/.zshrc

# Titans environment aliases
alias titans-activate="cd $(pwd) && ${USE_CONDA:+conda activate titans}${USE_CONDA:-source titans_env/bin/activate}"
alias titans-train="python -m titans_implementation.scripts.train"
alias titans-eval="python -m titans_implementation.scripts.evaluate"
alias titans-demo="python -m titans_implementation.scripts.demo"
alias titans-test="python -m titans_implementation.scripts.test"
EOT

echo "${fg[green]}Setup complete!${reset_color}"
echo "To activate Titans, run: ${fg[yellow]}source ~/.zshrc && titans-activate${reset_color}"
echo "You can now use these aliases:"
echo "  ${fg[yellow]}titans-train${reset_color} - Run the training script"
echo "  ${fg[yellow]}titans-eval${reset_color}  - Run the evaluation script"
echo "  ${fg[yellow]}titans-demo${reset_color}  - Run the demo script"
echo "  ${fg[yellow]}titans-test${reset_color}  - Run the test script" 