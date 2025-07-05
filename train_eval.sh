
################################################################## Dynamic Object
output=output/dynamic_object
exp=fan
#python train_gui.py -s /path/to/data/ -m $output/$exp --max_time 0.72 --light --physics_code 16
#python seg.py -m $output/$exp --K 2 --vis
#python render.py -m $output/$exp --mode render --skip_train
#python metrics.py -m $output/$exp --s test --half_res
#python metrics.py -m $output/$exp --s val --half_res

exp=whale
#python train_gui.py -s /path/to/data/ -m $output/$exp --max_time 0.75 --light --physics_code 16
#python render.py -m $output/$exp --mode render --skip_train
#python metrics.py -m $output/$exp --s test --half_res
#python metrics.py -m $output/$exp --s val --half_res

exp=shark
#python train_gui.py -s /path/to/data/ -m $output/$exp --max_time 0.75 --light --physics_code 16
#python render.py -m $output/$exp --mode render --skip_train
#python metrics.py -m $output/$exp --s test --half_res
#python metrics.py -m $output/$exp --s val --half_res

exp=telescope
#python train_gui.py -s /path/to/data/ -m $output/$exp --max_time 0.72 --light --physics_code 16
#python render.py -m $output/$exp --mode render --skip_train
#python metrics.py -m $output/$exp --s test --half_res
#python metrics.py -m $output/$exp --s val --half_res

exp=fallingball
#python train_gui.py -s /path/to/data/ -m $output/$exp --max_time 0.72 --light --physics_code 16
#python render.py -m $output/$exp --mode render --skip_train
#python metrics.py -m $output/$exp --s test --half_res
#python metrics.py -m $output/$exp --s val --half_res

exp=bat
#python train_gui.py -s /path/to/data/ -m $output/$exp --max_time 0.75 --light --physics_code 16
#python seg.py -m $output/$exp --K 3 --vis
#python render.py -m $output/$exp --mode render --skip_train
#python metrics.py -m $output/$exp --s test --half_res
#python metrics.py -m $output/$exp --s val --half_res
#python render.py -m $output/$exp --mode all --skip_val --skip_test
#

################################################################## ParticleNerf
output=output/particle_nerf
exp=pendulums
#python train_gui.py -s /path/to/data/ -m $output/$exp --max_time 0.75 --fps 60 --physics_code 16
#python render.py -m $output/$exp --mode render --skip_train --fps 60
#python metrics.py -m $output/$exp --s test --half_res
#python metrics.py -m $output/$exp --s val --half_res
#python render.py -m $output/$exp --mode all --skip_val --skip_test --fps 60

exp=cloth
#python train_gui.py -s /path/to/data/ -m $output/$exp --max_time 0.75 --fps 60 --physics_code 16
#python render.py -m $output/$exp --mode render --skip_train --fps 60
#python metrics.py -m $output/$exp --s test --half_res
#python metrics.py -m $output/$exp --s val --half_res
#python render.py -m $output/$exp --mode all --skip_val --skip_test --fps 60

exp=spring
#python train_gui.py -s /path/to/data/ -m $output/$exp --max_time 0.75 --fps 60 --physics_code 16
#python render.py -m $output/$exp --mode render --skip_train --fps 60
#python metrics.py -m $output/$exp --s test --half_res
#python metrics.py -m $output/$exp --s val --half_res
#python render.py -m $output/$exp --mode all --skip_val --skip_test --fps 60

exp=robot
#python train_gui.py -s /path/to/data/ -m $output/$exp --max_time 0.75 --fps 30 --physics_code 16
#python render.py -m $output/$exp --mode render --skip_train --fps 30
#python metrics.py -m $output/$exp --s test --half_res
#python metrics.py -m $output/$exp --s val --half_res
#python render.py -m $output/$exp --mode all --skip_val --skip_test

exp=robot-task
#python train_gui.py -s /path/to/data/ -m $output/$exp --max_time 0.75 --fps 30 --physics_code 16
#python render.py -m $output/$exp --mode render --skip_train --fps 30
#python metrics.py -m $output/$exp --s test --half_res
#python metrics.py -m $output/$exp --s val --half_res
#python render.py -m $output/$exp --mode all --skip_val --skip_test

exp=wheel
#python train_gui.py -s /path/to/data/ -m $output/$exp --max_time 0.75 --fps 60 --physics_code 16
#python render.py -m $output/$exp --mode render --skip_train --fps 60
#python metrics.py -m $output/$exp --s test --half_res
#python metrics.py -m $output/$exp --s val --half_res
#python render.py -m $output/$exp --mode all --skip_val --skip_test



#################################################################### Dynamic Indoor Scene
output=output/dynamic_indoor_scenes
exp=chessboard
#python train_gui.py -s /path/to/data/ -m $output/$exp --max_time 0.70 --light --physics_code 32
#python seg.py -m $output/$exp --K 10 --smooth 0.1 --vis --scene others
#python render.py -m $output/$exp --mode render --skip_train
#python metrics.py -m $output/$exp --s test
#python metrics.py -m $output/$exp --s val

exp=darkroom
#python train_gui.py -s /path/to/data/ -m $output/$exp --max_time 0.70 --light --light --physics_code 32
#python seg.py -m $output/$exp --K 6 --smooth 0.1 --vis --scene others
#python render.py -m $output/$exp --mode render --skip_train
#python metrics.py -m $output/$exp --s test
#python metrics.py -m $output/$exp --s val

exp=dining
#python train_gui.py -s /path/to/data/ -m $output/$exp --max_time 0.70 --light --light --physics_code 32
#python seg.py -m $output/$exp --K 10 --smooth 0.8 --vis --scene dining
#python render.py -m $output/$exp --mode render --skip_train
#python metrics.py -m $output/$exp --s test
#python metrics.py -m $output/$exp --s val

exp=factory
#python train_gui.py -s /path/to/data/ -m $output/$exp --max_time 0.70 --light --light --physics_code 32
#python seg.py -m $output/$exp --K 6 --smooth 0.5 --vis --scene others
#python render.py -m $output/$exp --mode render --skip_train
#python metrics.py -m $output/$exp --s test
#python metrics.py -m $output/$exp --s val

################################################################## NVIDIA
output=output/NVIDIA_dynamic_scene
exp=Skating
#python train_gui.py -s /path/to/data/ -m $output/$exp --max_time 0.65 --fps 20 --vel_start_time 0.35 --physics_code 16
##python seg.py -m $output/$exp --K 4 --vis
#python render.py -m $output/$exp --mode render --skip_train --fps 20
#python metrics.py -m $output/$exp --s test
#python metrics.py -m $output/$exp --s val

exp=Truck
#python train_gui.py -s /path/to/data/ -m $output/$exp --max_time 0.65 --fps 60 --physics_code 16
#python render.py -m $output/$exp --mode render --skip_train --fps 60
#python metrics.py -m $output/$exp --s test
#python metrics.py -m $output/$exp --s val

################################################################## GoPro
output=output/GOPRO
exp=pen1
#python train_gui.py -s /path/to/data/ -m $output/$exp --max_time 0.70 --physics_code 16
#python render.py -m $output/$exp --mode render --skip_train
#python metrics.py -m $output/$exp --s test
#python metrics.py -m $output/$exp --s val
#python render.py -m $output/$exp --mode original --skip_val --skip_test

exp=box
#python train_gui.py -s /path/to/data/ -m $output/$exp --max_time 0.70 --physics_code 16
#python render.py -m $output/$exp --mode render --skip_train
#python metrics.py -m $output/$exp --s test
#python metrics.py -m $output/$exp --s val

exp=mat
#python train_gui.py -s /path/to/data/ -m $output/$exp --max_time 0.75 --physics_code 16
#python render.py -m $output/$exp --mode render --skip_train --fps 88
#python metrics.py -m $output/$exp --s test
#python metrics.py -m $output/$exp --s val

exp=collision
#python train_gui.py -s /path/to/data/ -m $output/$exp --max_time 0.73 --physics_code 16
#python render.py -m $output/$exp --mode render --skip_train
#python metrics.py -m $output/$exp --s test
#python metrics.py -m $output/$exp --s val
#python render.py -m $output/$exp --mode original --skip_val --skip_test

exp=cube
#python train_gui.py -s /path/to/data/ -m $output/$exp --max_time 0.72 --physics_code 16
#python render.py -m $output/$exp --mode render --skip_train
#python metrics.py -m $output/$exp --s test
#python metrics.py -m $output/$exp --s val

exp=hammer
#python train_gui.py -s /path/to/data/ -m $output/$exp --max_time 0.70 --physics_code 16
#python render.py -m $output/$exp --mode render --skip_train
#python metrics.py -m $output/$exp --s test
#python metrics.py -m $output/$exp --s val

exp=pen2
#python train_gui.py -s /path/to/data/ -m $output/$exp --max_time 0.70 --physics_code 16
#python render.py -m $output/$exp --mode render --skip_train
#python metrics.py -m $output/$exp --s test
#python metrics.py -m $output/$exp --s val
