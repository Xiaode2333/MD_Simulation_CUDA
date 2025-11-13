// g++ ./tests/md_config_load_save/test_config.cpp \
    ./src/md_config.cpp \
    -o ./tests/md_config_load_save/test_config \
    -I${CONDA_PREFIX}/include -L${CONDA_PREFIX}/lib -Wl,-rpath,${CONDA_PREFIX}/lib \
    -lfmt
#include "../../include/md_config.hpp"

int main(){
    MDConfig cfg = {
        1000,
        1000,
        100.0,
        20.0,
        0.5,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.25,
        1.0,
        1.0,
        0.5,
        1e-3,
        100.0,
        0.1,
        "test",
        "loadpath",

        8,
        256,
    };

    MDConfigManager cfg_mng = MDConfigManager(cfg);
    cfg_mng.config_to_json("./tests/md_config_load_save/test_config_output.json");
    MDConfigManager loaded_cfg_mng = MDConfigManager::config_from_json("./tests/md_config_load_save/test_config_output.json"); 

    loaded_cfg_mng.print_config();

    return 0;
}
