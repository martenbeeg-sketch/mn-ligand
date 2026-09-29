# New Ubuntu Installation and Workstation Migration Protocol

This protocol installs mn-ligand on another Ubuntu workstation and migrates an
existing runtime archive and Docker images over a trusted local network. The
examples use:

- source workstation: `user@pc002975`
- destination workstation: `compute@NEW_HOST`
- destination repository: `/home/compute/programs/ovo-ligand`
- destination data pool: `/home/compute/data`
- destination app home: `/home/compute/data/mn-ligand`

Replace `NEW_HOST`, GPU IDs, and staging paths before running the commands. Do
not run source and destination workers against the same writable runtime. Keep
the source archive unchanged until the destination has passed verification and
a representative workflow has completed.

## Controlled migration in six checkpoints

The app is installed from GitHub on the destination. Runtime data is exported
to a separate portable copy and transferred independently. Nothing in this
procedure uninstalls or rewrites the current app, Conda environment, settings,
or source archive on this workstation. After export, the source installation
can continue using its own archive while the destination uses its own copy.

Pause at each checkpoint and inspect the result before continuing:

1. Prepare Ubuntu, Docker/NVIDIA support, Conda, SSH, and the mounted data pool
   on the destination. Do not share the source runtime directory over the
   network.
2. Clone the GitHub repository and create the Conda environment on the
   destination.
3. On the source, let queued/running jobs finish, then create and verify a
   portable export in a new staging directory. This reads the original archive
   and writes a separate copy.
4. Inspect the export report and its size. If it is satisfactory, start the
   source workers again and keep working on this computer if desired.
5. Use `rsync` to copy the verified export to the destination. Run destination
   verification and inspect several historical jobs in the app.
6. Transfer or build the Docker images needed there, then decide when to start
   destination workers and submit new jobs.

The source and destination archives are independent copies after step 5.
Changes made on one machine are not synchronized to the other. Do not configure
both installations to write to one shared runs directory.

If you want to stop after preparing GitHub on the destination, do so. The
source installation remains usable throughout, and data export can happen
later.

## What is and is not transferred

The migration has four independent parts:

1. application source code, installed from Git;
2. the Conda environments, recreated from the committed YAML files;
3. runtime data, copied from a verified portable export;
4. Docker images, copied with all local tags that refer to the required images.

The portable runtime contains `workdir/runs`, `reference_files`, and
`libraries`. It deliberately excludes temporary files and machine-local
configuration. The destination gets a new installation record and runtime
configuration.

Do not copy SSH keys, the MODELLER license key, Conda environment directories,
Docker's `/var/lib/docker`, systemd unit files, or the source workstation's
`~/.config/mn-ligand/installation.json`. Recreate those items locally.

Some images, model parameters, and weights have restricted licenses. Transfer
them only when the license permits use on the destination machine. In
particular, AlphaFold parameters, MODELLER credentials, and locally permissioned
generation models are not made redistributable by this protocol.

## 1. Preflight both workstations

Confirm that the machines can resolve and reach each other. On the source:

```bash
getent hosts NEW_HOST
ping -c 3 NEW_HOST
ssh compute@NEW_HOST 'hostname; id; df -h /home/compute/data'
```

Verify the SSH host-key fingerprint out of band before accepting it. Configure
an SSH key if desired; do not copy the source workstation's private key.

On the destination, install the basic transfer utilities:

```bash
sudo apt update
sudo apt install -y git rsync openssh-client openssh-server zstd
sudo systemctl enable --now ssh
```

The destination must also have:

- a compatible NVIDIA driver for its GPU;
- Docker Engine with the Compose plugin;
- NVIDIA Container Toolkit configured for Docker;
- a Conda-compatible installation such as Miniforge or Mambaforge;
- enough free space for the runtime, Docker images, and temporary transfer
  archives.

Use the current official Docker and NVIDIA installation instructions for the
Ubuntu release on the destination. Verify the runtime before copying anything:

```bash
nvidia-smi
docker version
docker compose version
docker info | sed -n '/Runtimes:/p'
conda --version
```

Ensure that the mergerfs pool is mounted and writable as the `compute` user:

```bash
findmnt --target /home/compute/data
test -w /home/compute/data
touch /home/compute/data/.mn-ligand-write-test
rm /home/compute/data/.mn-ligand-write-test
```

The `--require-mount /home/compute/data` setting used later makes the app and
workers refuse to start if this pool is not mounted.

## 2. Install the application on the destination

Run as `compute` on the destination:

```bash
mkdir -p /home/compute/programs
git clone git@github.com:martenbeeg-sketch/mn-ligand.git \
  /home/compute/programs/ovo-ligand
cd /home/compute/programs/ovo-ligand
git switch main
git pull --ff-only origin main

conda env create -f environment.yml
conda run -n mn-ligand python -m pip install -e .
conda run -n mn-ligand mn-ligand --help
```

If `mn-ligand` already exists, update it instead:

```bash
cd /home/compute/programs/ovo-ligand
git pull --ff-only origin main
conda env update -n mn-ligand -f environment.yml --prune
conda run -n mn-ligand python -m pip install -e .
```

### Optional MODELLER environment

Only create this environment if MODELLER-backed repair is needed. Enter the
license key on the destination; never transfer or commit it:

```bash
cd /home/compute/programs/ovo-ligand
read -rsp "MODELLER license key: " KEY_MODELLER
echo
export KEY_MODELLER
CONDA_CHANNEL_PRIORITY=strict conda env create -f environment-modeller.yml
unset KEY_MODELLER
conda run -n mn-ligand-modeller \
  python -c 'import modeller; print(modeller.__version__)'
```

The app discovers a sibling `mn-ligand-modeller` environment automatically.

Do not initialize the destination app home yet if the data copy will populate
that directory. It should be absent or empty before the first transfer.

## 3. Quiesce and audit the source runtime

First allow active simulations and analyses to reach a terminal state. A
portable export refuses to proceed while jobs are queued, running, or paused.
Then run on the source:

```bash
mn-ligand worker-service status --gpu-ids 0,1
mn-ligand worker-service stop --gpu-ids 0,1
mn-ligand portability audit
mn-ligand portability audit --json > /tmp/mn-ligand-portability-audit.json
```

Replace `0,1` with the source machine's configured GPU IDs. Leave the source
workers stopped through the final data copy. The audit is read-only.

Create the export on a filesystem with enough free space. The destination must
not already exist:

```bash
EXPORT_DIR=/path/with/free-space/mn-ligand-portable-$(date +%F)
test ! -e "$EXPORT_DIR"
mn-ligand portability export "$EXPORT_DIR"
mn-ligand portability verify "$EXPORT_DIR"
du -sh "$EXPORT_DIR"
```

The export preserves the source archive, converts operational metadata to
portable references in the copy, validates declared artifacts and checksums,
and writes `portable-export.json` plus `migration-report.json`.

## 4. Copy the runtime with rsync

`rsync` is recommended for a large runtime because interrupted copies can be
resumed. From the source:

```bash
ssh compute@NEW_HOST \
  'if test -e /home/compute/data/mn-ligand; then
     test -d /home/compute/data/mn-ligand &&
     ! find /home/compute/data/mn-ligand -mindepth 1 -print -quit | grep -q .
   fi'

rsync -a --partial --append-verify --info=progress2 \
  "$EXPORT_DIR/" \
  compute@NEW_HOST:/home/compute/data/mn-ligand/
```

The trailing slash on `"$EXPORT_DIR/"` is intentional: it copies the contents
into the destination app home. Do not add `--delete`; the protocol never
deletes destination data automatically.

Run the same command again after an interruption. A completed second run should
transfer little or no data. For a small bundle, `scp` is acceptable but is less
convenient to resume:

```bash
scp -r "$EXPORT_DIR" compute@NEW_HOST:/home/compute/data/
```

Do not use both commands for the same destination layout without checking the
resulting directory names.

If ownership is wrong because an administrator performed the copy, correct
only the dedicated application directory on the destination:

```bash
sudo chown -R compute:compute /home/compute/data/mn-ligand
```

## 5. Transfer the required Docker images

Building from the committed Compose file is the most reproducible option when
all build contexts and separately licensed inputs are available:

```bash
cd /home/compute/programs/ovo-ligand
../mn-tool-containers/build.sh mn-ligand
```

For large or locally customized images, transferring the tested source images
is faster and preserves the exact layers. The following source-side commands
combine Compose image names with registry image names and then include every
local alias attached to their image IDs. Thus both `latest` and a dated stable
tag are transferred when both point to the same image. Docker stores shared
layers only once after loading.

```bash
set -Eeuo pipefail
cd /home/user/programs/git-projects/ovo-ligand
IMAGE_STAGE=/path/with/free-space/mn-ligand-image-transfer
mkdir -p "$IMAGE_STAGE"

{
  docker compose config --images
  python3 - <<'PY'
import json
from pathlib import Path

payload = json.loads(Path("mn_ligand/manifests/tools.json").read_text())
for tool in payload["tools"]:
    image = str(tool.get("image") or "").strip()
    if image:
        print(image)
PY
} | sed '/^$/d' | sort -u > "$IMAGE_STAGE/required-images.txt"

UNSORTED_TAGS="$IMAGE_STAGE/transfer-image-tags.unsorted.txt"
: > "$UNSORTED_TAGS"
missing_images=0
while IFS= read -r image; do
  if ! docker image inspect "$image" >/dev/null 2>&1; then
    echo "Missing required image: $image" >&2
    missing_images=1
    continue
  fi
  docker image inspect \
    --format '{{range .RepoTags}}{{println .}}{{end}}' "$image" \
    >> "$UNSORTED_TAGS"
done < "$IMAGE_STAGE/required-images.txt"
if (( missing_images != 0 )); then
  exit 1
fi
sed '/^$/d' "$UNSORTED_TAGS" | sort -u \
  > "$IMAGE_STAGE/transfer-image-tags.txt"
rm "$UNSORTED_TAGS"

mapfile -t TRANSFER_TAGS < "$IMAGE_STAGE/transfer-image-tags.txt"
docker image save "${TRANSFER_TAGS[@]}" \
  | zstd -T0 -6 -o "$IMAGE_STAGE/mn-ligand-images.tar.zst"

(
  cd "$IMAGE_STAGE"
  sha256sum mn-ligand-images.tar.zst > mn-ligand-images.tar.zst.sha256
)
```

The inventory step deliberately stops at the first missing required image.
Build that Compose service on the source or destination, then rerun the
inventory. For example:

```bash
docker compose build abfe rbfe
```

Do not silently omit a missing image if its workflow is required on the new
workstation.

Copy the compressed archive and its inventory to the destination:

```bash
rsync -a --partial --append-verify --info=progress2 \
  "$IMAGE_STAGE/" \
  compute@NEW_HOST:/home/compute/data/mn-ligand-image-transfer/
```

Then load it on the destination:

```bash
cd /home/compute/data/mn-ligand-image-transfer
sha256sum -c mn-ligand-images.tar.zst.sha256
zstd -dc mn-ligand-images.tar.zst | docker image load
docker image ls
```

When neither workstation has room for an archive, stream the images directly.
This is not resumable, so use it only on a reliable LAN:

```bash
mapfile -t TRANSFER_TAGS < "$IMAGE_STAGE/transfer-image-tags.txt"
docker image save "${TRANSFER_TAGS[@]}" | zstd -T0 -3 \
  | ssh compute@NEW_HOST 'zstd -dc | docker image load'
```

The compressed archive may be removed manually only after the destination
passes all verification steps. Removing the archive does not remove loaded
Docker images.

## 6. Initialize the destination runtime

Run on the destination from the new environment:

```bash
cd /home/compute/programs/ovo-ligand

conda run -n mn-ligand mn-ligand init \
  --app-home /home/compute/data/mn-ligand \
  --runs-dir /home/compute/data/mn-ligand/workdir/runs \
  --reference-dir /home/compute/data/mn-ligand/reference_files \
  --library-dir /home/compute/data/mn-ligand/libraries \
  --tmpdir /home/compute/data/mn-ligand/tmp \
  --require-mount /home/compute/data

conda run -n mn-ligand mn-ligand portability verify \
  /home/compute/data/mn-ligand
```

This creates machine-local configuration for the new paths. It does not modify
the source workstation or its archive. Historical absolute paths remain in
immutable provenance where appropriate; operational references are resolved
against the destination roots.

Install activation-free launchers:

```bash
conda run -n mn-ligand mn-ligand install-launchers
```

After this, `/home/compute/.local/bin/mn-ligand` and `mn-ligand-app` work
without activating Conda. Ensure `/home/compute/.local/bin` is in `PATH`, or
invoke the wrappers by their absolute paths.

## 7. Validate before starting workers

Run the complete installation diagnostic on the destination:

```bash
/home/compute/.local/bin/mn-ligand doctor
/home/compute/.local/bin/mn-ligand doctor --json \
  > /home/compute/data/mn-ligand/destination-doctor.json
```

Review every failure and warning. At minimum, verify:

```bash
findmnt --target /home/compute/data
nvidia-smi
docker run --rm --gpus all --entrypoint nvidia-smi \
  mn-gromacs:2026.3-cu128
test -r /home/compute/data/mn-ligand/portable-export.json
test -d /home/compute/data/mn-ligand/workdir/runs
test -d /home/compute/data/mn-ligand/reference_files
test -d /home/compute/data/mn-ligand/libraries
```

Inspect several historical campaigns in the app before enabling execution:

```bash
/home/compute/.local/bin/mn-ligand-app
```

Confirm that job lists, structures, trajectories, reports, and cross-job links
open from the new paths. Do not mark historical failed jobs successful merely
because their files were copied.

## 8. Install and start destination workers

Choose GPU IDs from `nvidia-smi -L`. The following example installs one CPU
worker and GPU workers 0 and 1:

```bash
sudo loginctl enable-linger compute
/home/compute/.local/bin/mn-ligand worker-service install --gpu-ids 0,1
/home/compute/.local/bin/mn-ligand worker-service status --gpu-ids 0,1
```

User lingering lets the worker services start at boot without an interactive
login. Follow worker logs with:

```bash
journalctl --user \
  -u 'mn-ligand-worker@*.service' \
  -u mn-ligand-cpu-worker.service -f
```

Submit one small, non-critical representative job for each required execution
class before treating the migration as complete: one CPU job, one GPU job, and
one workflow that reads a migrated reference/model. Confirm native artifacts,
normalized results, and UI display.

## 9. Cutover and rollback

The migration is accepted only when all of the following are true:

- portable verification passes on the destination;
- `mn-ligand doctor` has no unexplained failures;
- required Docker images and external references are present;
- historical results can be opened;
- destination CPU and GPU workers are healthy;
- representative new jobs complete successfully.

Until then, keep the source runtime and Docker images unchanged. If validation
fails, stop destination workers and return to the source installation:

```bash
/home/compute/.local/bin/mn-ligand worker-service stop --gpu-ids 0,1
```

Once the destination is accepted, choose one authoritative workstation for new
job submission. Do not restart the source workers if that could create two
diverging archives. Retain the source as a read-only rollback copy until a
separate backup of the destination has been verified.

## Updating after migration

Application updates do not require another data migration:

```bash
cd /home/compute/programs/ovo-ligand
git pull --ff-only origin main
conda env update -n mn-ligand -f environment.yml --prune
conda run -n mn-ligand python -m pip install -e .
conda run -n mn-ligand mn-ligand install-launchers
conda run -n mn-ligand mn-ligand worker-service install --gpu-ids 0,1
```

Reinstalling launchers and worker units is important because they record the
current interpreter, checkout, app home, and temporary directory. Transfer or
rebuild Docker images separately only when their definitions or required tags
change.
