import math
import random


def cost(sequence, dist_matrix, index_to_position_map):
	"""计算序列的代价函数：任意相邻四个学生的距离之和"""
	cost_value = 0
	for i in range(len(sequence) - 3):
		for j in range(i, i + 4):
			for k in range(j + 1, i + 4):
				idx1 = index_to_position_map[sequence[j]]
				idx2 = index_to_position_map[sequence[k]]
				cost_value += dist_matrix[idx1][idx2]
	return cost_value


def simulated_annealing_updated(
	dist_matrix,
	init_sequence,
	index_to_position_map,
	init_temp=100,
	end_temp=0.1,
	alpha=0.995,
	max_iters=10000,
):
	"""模拟退火算法优化"""
	current_sequence = init_sequence.copy()
	current_cost = cost(current_sequence, dist_matrix, index_to_position_map)
	best_sequence = current_sequence
	best_cost = current_cost

	temp = init_temp

	for _ in range(max_iters):
		# 生成新的解决方案：交换两个随机位置的学生
		new_sequence = current_sequence.copy()
		idx1, idx2 = random.sample(range(len(new_sequence)), 2)
		new_sequence[idx1], new_sequence[idx2] = new_sequence[idx2], new_sequence[idx1]

		new_cost = cost(new_sequence, dist_matrix, index_to_position_map)

		# 如果新的代价函数值更低，或者满足一定的概率准则，接受新的序列
		if new_cost < current_cost or random.random() < math.exp((current_cost - new_cost) / temp):
			current_sequence = new_sequence
			current_cost = new_cost
			# 更新最佳序列
			if current_cost < best_cost:
				best_cost = current_cost
				best_sequence = current_sequence

		# 降低温度
		temp *= alpha
		if temp < end_temp:
			break

	return best_sequence
