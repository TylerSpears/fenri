NOTE: Due to the way that conda performs its environment switching, you must *source* the bash file, not execute it. For example:

```bash
# Do this
source export_conda_env.sh

# Don't do this!
# ./export_conda_env.sh
```

If you want to mass-upgrade packages, then use the `upgrader_plain_package_*` files along with the command

```bash
mamba upgrade --file upgrader_plain_package_environment.yml
pip install -U -r upgrader_plain_package_requirements.txt --upgrade-strategy eager
```
**NOTE** This hasn't been tested extensively, so be careful!
