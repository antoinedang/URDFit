watch -n 0.1 "echo ======PROGRESS====== && (cat data/SAC/0_steps/train.progress || true) && echo '' && echo =======OUT======= && (tail slurm/out.txt || true) && echo '' && echo =======ERROR======= && tail slurm/err.txt"
