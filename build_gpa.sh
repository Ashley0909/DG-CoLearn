cd partition
rm _partition.cpython-311-darwin.so
python3 setup.py build_ext --inplace
suffix=$(python3 - << 'EOF'
import sysconfig
print(sysconfig.get_config_var("EXT_SUFFIX"))
EOF
)
mv "partition${suffix}" "_partition${suffix}"
cd ..