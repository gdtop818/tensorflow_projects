#use this script to generate noise-corrupted test data
#the script add-noise-mod.py is used to achieve the noise injection
#refer to the info in add-noise-mod.py

noise_prior_cafe="0.0,0.0,0.0,10.0" #define noise type to sample. [S_clean, S_white, S_car, S_cafe]
noise_prior_car="0.0,0.0,10.0,0.0" #define noise type to sample. [S_clean, S_white, S_car, S_cafe]
noise_prior_white="0.0,10.0,0.0,0.0" #define noise type to sample. [S_clean, S_white, S_car, S_cafe]

noise_level=0
sigma0=0 #ensure the SNR is sampled as the value exacted defined by noise_level
seed=32
wav_scp=./wav.scp
verbose=0

declare -A noise_prior_box=(["white"]=$noise_prior_white ["car"]=$noise_prior_car ["cafe"]=$noise_prior_cafe)

for noise_type in white car cafe;do

noise_prior="${noise_prior_box["$noise_type"]}"
output_dir=../train_DAE/${noise_level}db/$noise_type
#echo $output_dir
#echo $noise_prior

mkdir -p $output_dir
./add-noise-mod.py --noise-level $noise_level --sigma0 $sigma0 --seed $seed --verbose $verbose --noise-prior=$noise_prior --noise-src noise.scp --wav-src $wav_scp --wavdir $output_dir

done
