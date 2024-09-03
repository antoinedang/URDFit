from stable_baselines3.common.callbacks import BaseCallback


class STDOutLogCallback(BaseCallback):
    def __init__(self, log_dir, num_envs, total_timesteps):
        super().__init__(0)
        self.num_envs = num_envs
        self.total_timesteps = total_timesteps
        self.log_dir = log_dir

    def _log_progress(self):
        steps_taken = self.num_envs * self.n_calls
        percent_progress = (steps_taken / self.total_timesteps) * 100
        with open(self.log_dir + "/train.progress", "w+") as f:
            f.write(
                "{}% ({} / {})\n".format(
                    percent_progress, steps_taken, self.total_timesteps
                )
            )

    def _on_step(self):
        self._log_progress()

        return True
