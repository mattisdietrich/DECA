# slurm
function _goto_triton(){
	ssh triton "cd $(pwd); $@";
}

if [[ $(hostname) != triton ]]; then
	alias scontrol="_goto_triton scontrol";
	alias sinfo="_goto_triton sinfo";
	alias squeue="_goto_triton squeue";
	alias srun="_goto_triton srun";
	alias sbatch="_goto_triton sbatch";
	alias salloc="_goto_triton salloc";
	alias scancel="_goto_triton scancel";
  alias sshare="_goto_triton sshare";
fi