from build_learning_plan import *
import sys

args = sys.argv
buf = "" if (len(args) == 1) else f"_{args[1]}"

## ================ config ================


# $ make
# $ learn


# SH_TEXT = "/media/nagalab/SSD1.7TB/nagalab/osada_ws/concrete_compaction/KerasFramework-master_tf2/train/learning_plan.sh"
SH_TEXT = f"/workspace/train/learning_plan{buf}.sh"

## metric
# NETWOEK_NAME_0 = "e-unet_sphereface"
# NETWOEK_NAME = "e-unet_metric_classifier"

## normal
# NETWOEK_NAME_0 = "e-unet"
NETWOEK_NAME = "e-unet_4class"
# NETWOEK_NAME = "espnet_4class"
# NETWOEK_NAME = "e-unet_after"

## metric
LOAD_ID_0 = "use-AE-input_20221014_ssim_mse_flip_rotate_20221110_AutoLearning"
# LOAD_ID_1 = "adam_dropout_20220805_AutoLearning"
# LOAD_ID = lambda x: f"u-conv_{x}_20220924_AutoLearning"
# LOAD_ID_1 = "convdense_20220926_AutoLearning"
# SAVE_ID = lambda x: f"u-conv_{x}_20220924_AutoLearning"
# SAVE_ID_1 = "convdense_20220926_AutoLearning"

## normal
# LOAD_ID = lambda *x: f"{x[0]}_adam_dropout_{x[1]}_AutoLearning" # 20220927
# LOAD_ID_1 = lambda x: f"{x}_adam_dropout_20221004_AutoLearning"
# LOAD_ID_1 = lambda x: f"{x}_adam_dropout_20221011_AutoLearning"
# LOAD_ID_1 = lambda *x: f"{x[0]}_{x[1]}_flip_20221103_AutoLearning"

AE_ID = "e-unet_20221014_ssim_mse"
# AE_ID = "espnet_20230101_ssim_mse"
# LOAD_ID_2 = f"use-AE-input_{AE_ID}_rotate_limit-45-45_20221129_AutoLearning"
# LOAD_ID_2 = f"use-AE-input_{AE_ID}_rotate_limit-ud_20221129_AutoLearning"
# LOAD_ID_2 = f"use-AE-input_{AE_ID}_rotate_limit-ud-lr_20221129_AutoLearning"
LOAD_ID_2 = lambda *reduce_const: f"use-AE-input_{AE_ID}_flip_rotate_reduce-{str(reduce_const[0]).replace(',', '-')}-{reduce_const[1]}_20230106_AutoLearning"
# LOAD_ID_2 = f"use-AE-input_{AE_ID}_rotate_limit-90-90_20221129_AutoLearning"
LOAD_ID_3 = f"20221231_AutoLearning"
# LOAD_ID_2 = f"use-AE-input_{AE_ID}_flip_rotate_20221120_AutoLearning"
# LOAD_ID_2 = f"use-AE-input_{AE_ID}_flip-ud_rotate_20221110_AutoLearning"
# LOAD_ID_2 = f"use-AE-input_{AE_ID}_flip-lr_rotate_20221110_AutoLearning"

# NORM = "group_norm"
# NORM = "batch_instance_norm"
# NORM = "batch_renorm"
NORM = "batch_norm"

# OPT = "adabelief"
OPT = "adam"
# OPT = "rectifiedadam"

LOSS = "$cross"
# LOSS = "cross_entropy_ssim"
# LOSS = "cross_entropy_iou"

DROPOUT = 0.25

# EUNET_METRIC_MODE = "u-conv"


## ========================================


Shell.new_file(SH_TEXT)
rep = WithoutReplace(
	arcface=True,
	cosface=True,
	sphereface=True,
	only_classifier=True,
	fourclass=True,
	after=True,
)

plan = list()
plan.append(CatInBox())


# for fold in range(1, 6):
	# if (fold == 3): continue
for fold in [1, 2, 3, 4, 5]:
	break
	for info in [
		# ["CrossEntropyIoU", "cross_entropy_iou", "rgb", "default", 0, 0],
		# ["area-minmax", "$cross", "rgb", "area_minmax", 0, 0],
		# ["HSV", "$cross", "hsv", "default", 0, 1],
		# ["HSV_area-minmax", "$cross", "hsv", "area_minmax",  0, 1],

		# ["LBP", "$cross", "lbp", "default",  0, 1],
		# ["LBP_area-minmax", "$cross", "lbp", "area_minmax",  0, 1],
		# ["LBP-3c","$cross", "lbp-3c", "default",  0, 1],
		# ["LBP-3c_area-minmax", "$cross", "lbp-3c", "area_minmax",  0, 1],
	
		# [None, "$cross", "rgb", "default", 0, 0, "tanh"],
		# ["LBP-3c", "$cross", "lbp-3c", "default", 0, 0, "tanh"],
		# ["use-AE-input", "$cross", "rgb", "default", 1, 1, "tanh"],
		# ["use-AE-input", "$cross", "rgb", "default", 1, 0, "linear"],
		
		# ["HoG","$cross", "hog", "default",  0, 0, "linear"],
		# ["HoG-3c","$cross", "hog-3c", "default",  0, 0, "linear"],
		# ["HoG","$cross", "hog", "default",  0, 0, "tanh"],
		# ["HoG-3c","$cross", "hog-3c", "default",  0, 0, "tanh"],

		# ["gray-lbp", "$cross", "gray-lbp", "default", 0, 1, "linear"],
		# ["gray-lbp", "$cross", "gray-lbp", "default", 0, 0, "tanh"],

		# ["use-AE-input", "20221014_ssim_mse", "$cross", "rgb", "default", 1, 1, 1, 1, "linear", "includeAE-noise"],
		# ["use-AE-input", "20221014_ssim_mse", "$cross", "rgb", "default", 1, 0, 0, 1, "tanh", "includeAE-noise"],
		# ["use-AE-input", "20221025_skip-connection-0", "$cross", "rgb", "default", 1, 1, 1, 1, "linear", "includeAE-noise"],
		# ["use-AE-input", "20221025_skip-connection-0", "$cross", "rgb", "default", 1, 0, 0, 1, "tanh", "includeAE-noise"],
		# ["use-AE-input", "20221025_skip-connection-1", "$cross", "rgb", "default", 1, 1, 1, 1, "linear", "includeAE-noise"],
		# ["use-AE-input", "20221025_skip-connection-1", "$cross", "rgb", "default", 1, 0, 0, 1, "tanh", "includeAE-noise"],
		# ["use-AE-input", "20221025_skip-connection-2", "$cross", "rgb", "default", 1, 1, 1, 1, "linear", "includeAE-noise"],
		# ["use-AE-input", "20221025_skip-connection-2", "$cross", "rgb", "default", 1, 0, 0, 1, "tanh", "includeAE-noise"],

		["use-AE-input", "20221014-1", "$cross", "rgb", "default", 1, 0, 0, 1, "linear", "includeAE-noise"],
		["use-AE-input", "20221014_ssim_mse", "$cross", "rgb", "default", 1, 0, 0, 1, "linear", "includeAE-noise"],
		[None, "none", "$cross", "rgb", "default", 0, 0, 0, 1, "linear", "linear"],

		# ["use-AE-input", "20221014-1", "$cross", "rgb", "default", 1, 0, 0, 1, "tanh", "includeAE-noise"],
		# ["use-AE-input", "20221014_ssim_mse", "$cross", "rgb", "default", 1, 0, 0, 1, "tanh", "includeAE-noise"],
		# [None, "none", "$cross", "rgb", "default", 0, 0, 0, 1, "tanh", "linear"],
		]:

		load_id, id_date, loss, color_type, normalization, use_AE_input, train, test, le, noise_type, train_noise_type = info
		threeclasses_test = 0

		## 改行
		plan.append(Indention())
		plan.append(Indention())
		plan.append(Indention())
		
		## mesh train
		# if learn_mesh:
		# plan.append(Train(is_autotrain=True,
		# 				auto_train_acc_limit=95,
		# 				fold=fold,
		# 				network_name=rep.network_name(NETWOEK_NAME),
		# 				save_path="$mesh",
		# 				save_id=LOAD_ID(load_id),
		# 				is_use_fullframe=False,
		# 				is_use_fresh=False,
		# 				is_load_weight=False,
		# 				#   load_weight_path=f"{rep.network_name(NETWOEK_NAME_0)}_{LOAD_ID_0}_fold{fold}_540x540",
		# 				is_extend_luminance=False,
		# 				is_use_BCL=False,
		# 				loss=loss,
		# 				optimizer=OPT,
		# 				is_h5=False,
		# 				is_use_metric=False,
		# 				is_load_fullframe_weight=False,
		# 				output_layer_name="classifier",
		# 				is_fusion_face=False,
		# 				nullfication_metric=False,
		# 				dropout_const=DROPOUT,
		# 				label_smoothing=0,
		# 				norm=NORM,
		# 				use_attention=False,
		# 				classification="fourclasses",
		# 				multi_losses=False,
		# 				fourclasses_type="default",
		# 				eunet_metric_mode="conv_dense",
		# 				eunet_metric_subcontext="default",
		# 				color_type=color_type,
		# 				normalization=normalization))

			# load_id_mesh = f"{rep.network_name(NETWOEK_NAME)}_4class_{load_id}_fold{fold}"
			# load_id_mesh = f"{rep.network_name(NETWOEK_NAME)}_4class_{LOAD_ID(load_id)}_fold{fold}"

		# else:
		load_id_mesh = f"{rep.network_name(NETWOEK_NAME)}_4class_{LOAD_ID_0}_fold{fold}_576x576"

		if (load_id in ["use-AE-input"]):
			load_id = LOAD_ID_1(load_id, id_date) if load_id is not None else LOAD_ID_0
		else:
			load_id = LOAD_ID(load_id, id_date) if load_id is not None else LOAD_ID_0

		# pixel train
		if train:
			plan.append(Train(is_autotrain=True,
							auto_train_acc_limit=95,
							fold=fold,
							network_name=rep.network_name(NETWOEK_NAME),
							save_path="$fullframe",
							save_id=load_id,
							is_use_fullframe=True,
							is_use_fresh=False,
							is_load_weight=True,
							load_weight_path=load_id_mesh,
							is_extend_luminance=False,
							is_use_BCL=False,
							loss=loss,
							optimizer=OPT,
							is_h5=False,
							is_use_metric=False,
							is_load_fullframe_weight=True, # False,
							output_layer_name="classifier",
							is_fusion_face=False,
							nullfication_metric=False,
							dropout_const=DROPOUT,
							label_smoothing=0,
							norm=NORM,
							use_attention=False,
							classification="fourclasses",
							multi_losses=False,
							fourclasses_type="default",
							eunet_metric_mode="conv_dense",
							eunet_metric_subcontext="default",
							color_type=color_type,
							normalization=normalization,
							use_AE_input=use_AE_input,
							noise_type=train_noise_type,
							autoencoder_loss="ssim",
							AE_model_id=id_date))
			plan.append(Indention())

		# # test
		if test:
			for fourclasses_test in [0]:
				for judge_by_mesh in [0, 1]:
					plan.append(Test(fold=fold,
									network_name=rep.network_name(NETWOEK_NAME),
									load_path="$fullframe",
									load_id=f"{rep.network_name(NETWOEK_NAME)}_4class_{load_id}_fold{fold}",
									size=(576, 576),
									# size=(540, 540),
									is_use_fullframe=True,
									is_use_fresh=False,
									is_judgement_by_mesh=judge_by_mesh,
									is_fusion_face=False,
									norm=NORM,
									is_use_averageimage=True,
									use_attention=False,
									classification="fourclasses",
									is_quantized=False,
									do_fourclasses_test=fourclasses_test,
									multi_losses=False,
									fourclasses_type="default",
									is_use_LE=False,
									use_custom_loss=loss != "$cross",
									loss=loss,
									color_type=color_type,
									normalization=normalization,
									use_AE_input=use_AE_input,
									noise_type=noise_type,
									autoencoder_loss="ssim",
									AE_model_id=id_date,
									do_threeclasses_test=threeclasses_test))
			plan.append(Indention())

		## 輝度の拡張によるテスト
		if le:
			# for LE_mode in ["half", "slant", "circle", "all"]:
			for LE_mode in ["half", "circle", "all"]:
				plan.append(Indention())
				for LE_const in [100 - (25*i) for i in range(9)]:
					if not LE_const and (noise_type != "tanh"): continue
				
					plan.append(Test(fold=fold,
									network_name=rep.network_name(NETWOEK_NAME),
									load_path="$fullframe",
									load_id=f"{rep.network_name(NETWOEK_NAME)}_4class_{load_id}_fold{fold}",
									size=(576, 576),
									# size=(540, 540),
									is_use_fullframe=True,
									is_use_fresh=False,
									is_judgement_by_mesh=False,
									is_fusion_face=False,
									norm=NORM,
									is_use_averageimage=True,
									use_attention=False,
									classification="fourclasses",
									is_quantized=False,
									do_fourclasses_test=False,
									multi_losses=False,
									fourclasses_type="default",
									is_use_LE=True,
									LE_mode=LE_mode,
									LE_const=LE_const,
									use_custom_loss=loss != "$cross",
									loss=loss,
									color_type=color_type,
									normalization=normalization,
									use_AE_input=use_AE_input,
									noise_type=noise_type,
									autoencoder_loss="ssim",
									AE_model_id=id_date))


# dropout = 0.25

# ## 4 classes metric (u-conv)
# if 1:
# 	for fold in [3]:
# 		for eunet_metric_subcontext, name in zip(["4,5,6_2", "4,5,6_bottle", "5,6,7_2", "4,4,4,5,5_2", "4,4,4,4,5,5,5_2"],
#                                     			["456-2", "456-b", "567-2", "44455-2", "4444555-2"]):
		
# 			## 改行
# 			plan.append(Indention())
	
# 			# pixel metric train
# 			plan.append(Train(is_autotrain=True,
# 							auto_train_acc_limit=95,
# 							fold=fold,
# 							network_name=NETWOEK_NAME_0,
# 							save_path="$fullframe",
# 							save_id=SAVE_ID(name),
# 							is_use_fullframe=True,
# 							is_use_fresh=False,
# 							is_load_weight=True,
# 							load_weight_path=f"{rep.network_name(NETWOEK_NAME_0)}_4class_{LOAD_ID_0}_fold{fold}_576x576",
# 							is_extend_luminance=False,
# 							is_grayscale=False,
# 							is_use_BCL=False,
# 							loss="$cross",
# 							optimizer=OPT,
# 							is_h5=False,
			
# 							is_use_metric=True,
# 							metric_func="sphereface",
# 							is_load_fullframe_weight=True,
# 							output_layer_name="classifier",
# 							is_fusion_face=False,
# 							nullfication_metric=False,
# 							dropout_const=dropout,
# 							label_smoothing=0,
# 							norm=NORM,
# 							use_attention=False,
# 							classification="fourclasses",
# 							multi_losses=False,
# 							fourclasses_type="default",
# 							eunet_metric_mode=EUNET_METRIC_MODE,
# 							eunet_metric_subcontext=eunet_metric_subcontext))
	
	
# 			# pixel metric train
# 			plan.append(Train(is_autotrain=True,
# 							auto_train_acc_limit=95,
# 							fold=fold,
# 							network_name=NETWOEK_NAME,
# 							save_path="$fullframe",
# 							save_id=SAVE_ID(name),
# 							is_use_fullframe=True,
# 							is_use_fresh=False,
# 							is_load_weight=True,
# 							load_weight_path=f"{NETWOEK_NAME_0}_4class_{LOAD_ID(name)}_fold{fold}_576x576",
# 							is_extend_luminance=False,
# 							is_grayscale=False,
# 							is_use_BCL=False,
# 							loss="$cross",
# 							optimizer=OPT,
# 							is_h5=False,
			
# 							is_use_metric=True,
# 							metric_func="sphereface",
# 							is_load_fullframe_weight=True,
# 							output_layer_name="classifier",
# 							is_fusion_face=False,
# 							nullfication_metric=False,
# 							dropout_const=dropout,
# 							label_smoothing=0,
# 							norm=NORM,
# 							use_attention=False,
# 							classification="fourclasses",
# 							multi_losses=False,
# 							fourclasses_type="default",
# 							eunet_metric_mode=EUNET_METRIC_MODE,
# 							eunet_metric_subcontext=eunet_metric_subcontext))
			
# 			## test
# 			for fourclasses_test in [0, 1]:
# 				for judge_by_mesh in [0, 1]:
# 					plan.append(Test(fold=fold,
# 									network_name=NETWOEK_NAME_0,
# 									load_path="$fullframe",
# 									load_id=f"{NETWOEK_NAME}_4class_{LOAD_ID(name)}_fold{fold}",
# 									size=(576, 576),
# 									# size=(540, 540),
# 									is_use_fullframe=True,
# 									is_use_fresh=False,
# 									is_judgement_by_mesh=judge_by_mesh,
# 									is_grayscale=False,
# 									is_fusion_face=False,
# 									norm=NORM,
# 									is_use_averageimage=True,
# 									use_attention=False,
# 									classification="fourclasses",
# 									is_quantized=False,
# 									do_fourclasses_test=fourclasses_test,
# 									multi_losses=False,
# 									fourclasses_type="default"))
	

# plan.append(Indention())
# plan.append(Indention())

# fold = 1

for fold in range(1, 6):
    
	break

	## 改行
	plan.append(Indention())
		
	# pixel train
	plan.append(Train(is_autotrain=True,
					auto_train_acc_limit=95,
					fold=fold,
					network_name=rep.network_name(NETWOEK_NAME),
					save_path="$fullframe",
					save_id=LOAD_ID_2,
					is_use_fullframe=True,
					is_use_fresh=False,
					is_load_weight=False,
					load_weight_path=f"{rep.network_name(NETWOEK_NAME)}_4class_{LOAD_ID_0}_fold{fold}_576x576",
					is_extend_luminance=False,
					is_grayscale=False,
					is_use_BCL=False,
					loss=LOSS,
					optimizer=OPT,
					is_h5=False,
					is_use_metric=False,
					is_load_fullframe_weight=True,
					output_layer_name="classifier",
					is_fusion_face=False,
					nullfication_metric=False,
					dropout_const=0.25,
					label_smoothing=0,
					norm="batch_norm",
					use_attention=False,
					# classification="fourclasses",
					classification="before-just-after",
					multi_losses=False,
					fourclasses_type="default",
					eunet_metric_mode="conv_dense",
					eunet_metric_subcontext="default",
					color_type="rgb",
					normalization="default",
					noise_type="includeAE-noise",
					use_AE_input=True,
					AE_model_id=AE_ID,
					is_flip=True,
					is_rotate=True,
					is_enlarge=False,
					reduce_const=1,
    				learning_rate=1e-3,
					rotate_rate=0.90
	))
		
	# pixel test
	for fourclasses_test in range(1):
		for mesh_test in range(2):
			plan.append(Test(fold=fold,
							network_name=rep.network_name(NETWOEK_NAME),
							load_path="$fullframe",
							load_id=f"{rep.network_name(NETWOEK_NAME)}_after_{LOAD_ID_2}_fold{fold}",
							# load_id=f"{rep.network_name(NETWOEK_NAME)}_4class_{LOAD_ID_2}_fold{fold}",
							size=(576, 576),
							is_use_fullframe=True,
							is_use_fresh=False,
							is_judgement_by_mesh=mesh_test,
							is_grayscale=False,
							is_fusion_face=False,
							norm="batch_norm",
							is_use_averageimage=True,
							use_attention=False,
							# classification="fourclasses",
							classification="before-just-after",
							is_quantized=False,
							do_fourclasses_test=fourclasses_test,
							do_threeclasses_test=True,
							# do_threeclasses_test=False,
							multi_losses=False,
							fourclasses_type="default",
							is_use_LE=False,
							use_AE_input=True,
							AE_model_id=AE_ID))
   

## reduce weights

fold = 3

# for fold in [3]:

for reduce in ["ssim", "error", "zero", "square-error"]:
	# break
	# for reduce_const in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8][::-1]:
	for reduce_const in [0.75, 0.5, 0.25]:

		# break
	
		## 改行
		plan.append(Indention())
			
		# pixel train
		plan.append(Train(is_autotrain=True,
						auto_train_acc_limit=95,
						fold=fold,
						network_name=rep.network_name(NETWOEK_NAME),
						save_path="$fullframe",
						save_id=LOAD_ID_2(reduce_const, reduce),
						is_use_fullframe=True,
						is_use_fresh=False,
						is_load_weight=False,
						load_weight_path=f"{rep.network_name(NETWOEK_NAME)}_4class_{LOAD_ID_0}_fold{fold}_576x576",
						is_extend_luminance=False,
						is_grayscale=False,
						is_use_BCL=False,
						loss="$cross",
						optimizer=OPT,
						is_h5=False,
						is_use_metric=False,
						is_load_fullframe_weight=True,
						output_layer_name="classifier",
						is_fusion_face=False,
						nullfication_metric=False,
						dropout_const=0.01,
						label_smoothing=0,
						norm="batch_norm",
						use_attention=False,
						classification="fourclasses",
						# classification="before-just-after",
						multi_losses=False,
						fourclasses_type="default",
						eunet_metric_mode="conv_dense",
						eunet_metric_subcontext="default",
						color_type="rgb",
						normalization="default",
						noise_type="includeAE-noise",
						use_AE_input=True,
						AE_model_id=AE_ID,
						is_flip=True,
						flip_list=[0, 0, 1, 1, 2, 3],
						is_rotate=True,
						rotate_degrees=[[0, 360]],
						is_enlarge=False,
						reduce_const=reduce_const,
						learning_rate=1e-3,
						rotate_rate=0.75,
						all_in_one=False,
						reduce=reduce,
		))
			
		# pixel test
		for fourclasses_test in range(1):
			for mesh_test in range(1):
				plan.append(Test(fold=fold,
								network_name=rep.network_name(NETWOEK_NAME),
								load_path="$fullframe",
								load_id=f"{rep.network_name(NETWOEK_NAME)}_4class_{LOAD_ID_2(reduce_const, reduce)}_fold{fold}",
								size=(576, 576),
								is_use_fullframe=True,
								is_use_fresh=False,
								is_judgement_by_mesh=mesh_test,
								is_grayscale=False,
								is_fusion_face=False,
								norm="batch_norm",
								is_use_averageimage=True,
								use_attention=False,
								classification="fourclasses",
								# classification="before-just-after",
								is_quantized=False,
								do_fourclasses_test=fourclasses_test,
								# do_threeclasses_test=True,
								do_threeclasses_test=False,
								multi_losses=False,
								fourclasses_type="default",
								is_use_LE=False,
								use_AE_input=True,
								AE_model_id=AE_ID))
   

## chenge model
for fold in [3]:

	break
	## 改行
	plan.append(Indention())
		
	# mesh train
	plan.append(Train(is_autotrain=True,
					auto_train_acc_limit=95,
					fold=fold,
					network_name=rep.network_name(NETWOEK_NAME),
					save_path="$mesh",
					save_id=LOAD_ID_3,
					is_use_fullframe=False,
					is_use_fresh=False,
					is_load_weight=False,
					# load_weight_path=f"{rep.network_name(NETWOEK_NAME)}_4class_{LOAD_ID_3}_fold{fold}_576x576",
					is_extend_luminance=False,
					is_grayscale=False,
					is_use_BCL=False,
					loss="$cross",
					optimizer=OPT,
					is_h5=False,
					is_use_metric=False,
					is_load_fullframe_weight=False,
					output_layer_name="classifier",
					is_fusion_face=False,
					nullfication_metric=False,
					dropout_const=0.01,
					label_smoothing=0,
					norm="batch_norm",
					use_attention=False,
					classification="fourclasses",
					multi_losses=False,
					color_type="rgb",
					normalization="default",
					use_AE_input=False,
					is_flip=True,
					flip_list=[0, 0, 1, 1, 2, 3],
					is_rotate=True,
					rotate_degrees=[[0, 360]],
					is_enlarge=False,
					reduce_const=1,
					learning_rate=1e-3,
					rotate_rate=0.75,
					all_in_one=False,
	))
 
	# pixel train
	plan.append(Train(is_autotrain=True,
					auto_train_acc_limit=95,
					fold=fold,
					network_name=rep.network_name(NETWOEK_NAME),
					save_path="$fullframe",
					save_id=LOAD_ID_3,
					is_use_fullframe=True,
					is_use_fresh=False,
					is_load_weight=True,
					load_weight_path=f"{rep.network_name(NETWOEK_NAME)}_4class_{LOAD_ID_3}_fold{fold}",
					is_extend_luminance=False,
					is_grayscale=False,
					is_use_BCL=False,
					loss="$cross",
					optimizer=OPT,
					is_h5=False,
					is_use_metric=False,
					is_load_fullframe_weight=False,
					output_layer_name="classifier",
					is_fusion_face=False,
					nullfication_metric=False,
					dropout_const=0.01,
					label_smoothing=0,
					norm="batch_norm",
					use_attention=False,
					classification="fourclasses",
					# classification="before-just-after",
					multi_losses=False,
					fourclasses_type="default",
					eunet_metric_mode="conv_dense",
					eunet_metric_subcontext="default",
					color_type="rgb",
					normalization="default",
					noise_type="includeAE-noise",
					use_AE_input=True,
					AE_model_id=AE_ID,
					is_flip=True,
					flip_list=[0, 0, 1, 1, 2, 3],
					is_rotate=True,
					rotate_degrees=[[0, 360]],
					is_enlarge=False,
					reduce_const=1,
					learning_rate=1e-3,
					rotate_rate=0.75,
					all_in_one=False,
	))
		
	# pixel test
	for fourclasses_test in range(2):
		for mesh_test in range(1):
			plan.append(Test(fold=fold,
							network_name=rep.network_name(NETWOEK_NAME),
							load_path="$fullframe",
							load_id=f"{rep.network_name(NETWOEK_NAME)}_4class_{LOAD_ID_3}_fold{fold}",
							size=(576, 576),
							is_use_fullframe=True,
							is_use_fresh=False,
							is_judgement_by_mesh=mesh_test,
							is_grayscale=False,
							is_fusion_face=False,
							norm="batch_norm",
							is_use_averageimage=True,
							use_attention=False,
							classification="fourclasses",
							# classification="before-just-after",
							is_quantized=False,
							do_fourclasses_test=fourclasses_test,
							# do_threeclasses_test=True,
							do_threeclasses_test=False,
							multi_losses=False,
							fourclasses_type="default",
							is_use_LE=False,
							use_AE_input=True,
							AE_model_id=AE_ID))
   
   
## 4classes
for fold in range(1, 6):

	break
 
	## 改行
	plan.append(Indention())
		
	# pixel train
	plan.append(Train(is_autotrain=True,
					auto_train_acc_limit=95,
					fold=fold,
					network_name=rep.network_name(NETWOEK_NAME),
					save_path="$fullframe",
					save_id=LOAD_ID_2,
					is_use_fullframe=True,
					is_use_fresh=False,
					is_load_weight=True,
					load_weight_path=f"{rep.network_name(NETWOEK_NAME)}_4class_{LOAD_ID_1}_fold{fold}_576x576",
					is_extend_luminance=False,
					is_grayscale=False,
					is_use_BCL=False,
					loss="$cross",
					optimizer=OPT,
					is_h5=False,
					is_use_metric=False,
					is_load_fullframe_weight=True,
					output_layer_name="classifier",
					is_fusion_face=False,
					nullfication_metric=False,
					dropout_const=0.25,
					label_smoothing=0,
					norm="batch_norm",
					use_attention=False,
					classification="fourclasses",
					# classification="before-just-after",
					multi_losses=False,
					fourclasses_type="default",
					eunet_metric_mode="conv_dense",
					eunet_metric_subcontext="default",
					color_type="rgb",
					normalization="default",
					noise_type="includeAE-noise",
					use_AE_input=True,
					AE_model_id=AE_ID,
					is_flip=False,
					flip_list=[0, 0, 1, 1, 2, 3],
					is_rotate=True,
					rotate_degrees=[[0, 10], [80, 100], [170, 190], [260, 280], [350, 360]],
					is_enlarge=False,
					reduce_const=1,
    				learning_rate=1e-3,
					rotate_rate=0.5,
					all_in_one=False,
	))
		
	# pixel test
	for fourclasses_test in range(2):
		for mesh_test in range(1):
			plan.append(Test(fold=fold,
							network_name=rep.network_name(NETWOEK_NAME),
							load_path="$fullframe",
							# load_id=f"{rep.network_name(NETWOEK_NAME)}_after_{LOAD_ID_2}_fold{fold}",
							load_id=f"{rep.network_name(NETWOEK_NAME)}_4class_{LOAD_ID_2}_fold{fold}",
							size=(576, 576),
							is_use_fullframe=True,
							is_use_fresh=False,
							is_judgement_by_mesh=mesh_test,
							is_grayscale=False,
							is_fusion_face=False,
							norm="batch_norm",
							is_use_averageimage=True,
							use_attention=False,
							classification="fourclasses",
							# classification="before-just-after",
							is_quantized=False,
							do_fourclasses_test=fourclasses_test,
							# do_threeclasses_test=True,
							do_threeclasses_test=False,
							multi_losses=False,
							fourclasses_type="default",
							is_use_LE=False,
							use_AE_input=True,
							AE_model_id=AE_ID))



# plan.append(Indention("\npython /workspace/train/calc_correct_map.py"))
# plan.append(Indention())

tests = [
    # ["unet", lambda fold: f"unet_20220319_AutoLearning_fold{fold}", (540, 540), True, False],
    # ["unet", lambda fold: f"unet_batchRe_20220609_AutoLearning_foldd{fold}", (540, 540), False, True],
    # ["unet_metric_classifier", lambda fold: f"unet_metric_classifier_20220517_AutoLearning_fold{fold}", (540, 540), False, True],
    # ["unet_metric_classifier", lambda fold: f"unet_metric_classifier_dropout_20220530_AutoLearning_fold{fold}", (540, 540), False, True],
    ["e-unet", lambda fold: f"e-unet_4class_adam_dropout_20220805_AutoLearning_fold{fold}", (576, 576), False, True, "fourclasses", False, False],
    ["e-unet", lambda fold: f"e-unet_4class_adam_dropout_20220805_AutoLearning_fold{fold}", (576, 576), False, True, "fourclasses", False, True],
    # ["e-unet", lambda fold: f"e-unet_metric_classifier_4class_20220914_AutoLearning_fold{fold}", (576, 576), False, True, "fourclasses", False, True],
    # ["e-unet", lambda fold: f"e-unet_metric_classifier_4class_20220914_AutoLearning_fold{fold}", (576, 576), False, True, "fourclasses", False, False],
    # ["e-unet", lambda fold: f"e-unet_4class_20220723_AutoLearning_fold{fold}", (576, 576), False, True, "fourclasses", False, False],
    # ["e-unet", lambda fold: f"e-unet_4class_20220723_AutoLearning_fold{fold}", (576, 576), False, True, "fourclasses", False, True],
]


# if 0:
# 	fold = 3
# 	mesh_test = False
# 	for test in tests:
# 		network, load_id, size, is_h5, use_avg, clss, is_quat, C4test = test

# 		## 改行
# 		plan.append(Indention())

# 		# for fold in range(1, 6):
# 		for fold in [3]:
# 			## test

# 			## 改行
# 			plan.append(Indention())

# 			# for mesh_test in range(2):
# 			for LE_mode in ["half", "slant", "circle", "all"]:
# 				for LE_const in [100 - (25*i) for i in range(9)]:
# 					if not LE_const: continue
				
# 					plan.append(Test(fold=fold,
# 									network_name=network,
# 									load_path="$fullframe",
# 									load_id=load_id(fold),
# 									size=size,
# 									is_use_fullframe=True,
# 									is_use_fresh=False,
# 									is_judgement_by_mesh=mesh_test,
# 									is_grayscale=False,
# 									is_fusion_face=False,
# 									is_h5=is_h5,
# 									is_use_averageimage=use_avg,
# 									classification=clss,
# 									is_quantized=is_quat,
# 									do_fourclasses_test=C4test,
# 									multi_losses=False,
# 									fourclasses_type="default",
# 									is_use_LE=True,
# 									LE_mode=LE_mode,
# 									LE_const=LE_const))


for p in plan:
    p.output_shellscript(SH_TEXT, p.params)
