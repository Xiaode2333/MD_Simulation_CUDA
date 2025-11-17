module reset
module --force purge

module load OpenMPI
module load poetry
module load git
module load CUDA
module load CMake
module load nlohmann_json

module list

export VCPKG_CMAKE="$HOME/vcpkg/scripts/buildsystems/vcpkg.cmake"
export PYTHONPATH=$(poetry run python -c "import site; print(site.getsitepackages()[0])")

poetry install --no-root
poetry shell # Activate poetry env



