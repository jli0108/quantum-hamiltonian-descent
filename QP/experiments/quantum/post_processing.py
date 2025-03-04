import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds
import multiprocessing
from joblib import Parallel, delayed
from os.path import join
import sys
from time import time

# import data directory
sys.path.insert(1, '../../../')
from config import * 


def post_processing(benchmark_name, instance, resolution, device_label, alg):
	print(f"Run post-processing on instance No. {instance} from {benchmark_name}.")

	benchmark_dir = join(DATA_DIR_QP, benchmark_name)
	instance_dir = join(benchmark_dir, f"instance_{instance}")
	
	# Load instance data
	instance_filename = join(instance_dir, f"instance_{instance}.npy")
	with open(instance_filename, 'rb') as f:
		Q = np.load(f)
		b = np.load(f)
		Q_c = np.load(f)
		b_c = np.load(f)

	# Load sample data
	if alg == "sim_qhd" or alg == "sim_qaa":
		sample_filename = f"{device_label}_{alg}_rez{resolution}_T1000_sample_{instance}.npy"
	else:
		sample_filename = f"{device_label}_{alg}_rez{resolution}_sample_{instance}.npy"

	filename = join(instance_dir, sample_filename)
	samples = np.load(filename)
	numruns = len(samples)
	print(f'ID: {instance} -- Number of runs: {numruns}.')

	# Build the post-processing model
	dimension = len(Q)
	bounds = Bounds(np.zeros(dimension), np.ones(dimension))

	def qp_fun(x):
		return 0.5 * x @ Q @ x + b @ x

	def qp_der(x):
		return Q @ x + b

	post_samples = np.zeros((numruns, dimension))
	runtimes = []
	for k in range(numruns):
		x0 = samples[k]
		start_time = time()
		result = minimize(qp_fun, x0, method='TNC', jac=qp_der, bounds=bounds,
                            options={'gtol': 1e-9, 'eps': 1e-9})
		runtimes.append(time() - start_time)
		post_samples[k] = result.x
		if k % 100 == 0:
			print(f'ID: {instance} -- The {k}-th run has completed.')

	# Save post-processed samples
	post_sample_filename = "tnc_post_" + sample_filename
	post_filename = join(instance_dir, post_sample_filename)
	with open(post_filename , 'wb') as f:
		np.save(f, post_samples)
	
	if alg == "sim_qhd" or alg == "sim_qaa":
		qpu_time = 2e-8
		np.save(join(instance_dir, f"tnc_post_{device_label}_{alg}_rez{resolution}_T1000_runtime_{instance}.npy"), np.average(runtimes) + qpu_time)
	else:
		qpu_time = np.load(join(benchmark_dir, f"instance_{instance}/{device_label}_{alg}_rez{resolution}_runtime_{instance}.npy"))
		# Save the average runtime
		np.save(join(instance_dir, f"tnc_post_{device_label}_{alg}_rez{resolution}_runtime_{instance}.npy"), np.average(runtimes) + qpu_time)
		
	print(f"Benchmark: {benchmark_name}, instance: {instance}, post-processed sample saved.")

	return qpu_time


if __name__ == "__main__":
	dimension = 5
	sparsity = 5
	if dimension == 5:
		benchmark_name = f"QP-{dimension}d"
	else:
		benchmark_name = f"QP-{dimension}d-{sparsity}s"
	num_instances = 10
	resolution = 4
	alg = "sim_qaa"
	

	# device_name = "Advantage_system6.4"
	# device_name = "Advantage2_prototype2.6"
	device_name = "Advantage_system7.1"

	if device_name == "Advantage_system6.4":
		device_label = "adv_6.4"
	elif device_name == "Advantage2_prototype2.6":
		device_label = "adv2_2.6"
	elif device_name == "Advantage_system7.1":
		device_label = "adv_7.1"

	num_cores = multiprocessing.cpu_count()
	print(f'Num. of cores: {num_cores}.')

	par_list = Parallel(n_jobs=num_cores)(delayed(post_processing)(benchmark_name, tid, resolution, device_label, alg) for tid in range(num_instances))
