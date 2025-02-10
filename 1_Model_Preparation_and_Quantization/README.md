1. Please active vitis HLS
2. Revise batch_size if cuda got error
3. Add extra swap space on your Linux system
sudo swapoff -a
sudo rm /swapfile
sudo fallocate -l 64G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile

sudo swapon /swapfile
free -h


vitis_hls -f build_prj.tcl csim=0 synth=1 cosim=0 export=1
Here’s what each parameter means:

csim=0: Disables the C simulation step.
synth=1: Runs synthesis to convert the C/C++ design into RTL.
cosim=0: Disables cosimulation, which normally checks that the synthesized RTL matches the C simulation results.
export=1: Exports the final RTL (or IP core) project.