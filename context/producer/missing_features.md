# producer/ — Missing Features

- No Dockerized deployment spec; must run manually or add a K8s Job / CronJob
- No rate configurability via CLI; rate is hardcoded (0.01s interval = 100 msg/s)
- No built-in drift trigger (e.g., switch from normal to data_drift after N messages via signal or config); must restart with different trend argument
- No metrics / health endpoint for monitoring producer throughput
