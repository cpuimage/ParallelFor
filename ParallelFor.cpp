#include <stdio.h>
#include <stdlib.h>   
#include <iostream>


#if defined(_OPENMP)
// compile with: /openmp  
#include <omp.h>
auto const epoch = omp_get_wtime();
double now() {
	return omp_get_wtime() - epoch;
};
#else 
#include <chrono>
auto const epoch = std::chrono::steady_clock::now();
double now() {
	return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - epoch).count() / 1000.0;
};
#endif

template<typename FN>
double bench(const FN &fn) {
	auto took = -now();
	return (fn(), took + now());
}

#include <functional>

#if defined(_OPENMP)
#	include <omp.h>
#else 
#include <thread>

#include <vector>
#endif


#ifdef _OPENMP
static int processorCount = static_cast<int>(omp_get_num_procs());
#else
static int processorCount = static_cast<int>(std::thread::hardware_concurrency());
#endif


static void ParallelFor(int inclusiveFrom, int exclusiveTo, std::function<void(size_t)> func)
{
#if defined(_OPENMP)
#pragma omp parallel for num_threads(processorCount)
	for (int i = inclusiveFrom; i < exclusiveTo; ++i)
	{
		func(i);
	}
	return;
#else  
	if (inclusiveFrom >= exclusiveTo)
		return;

	static	size_t thread_cnt = 0;
	if (thread_cnt == 0)
	{
		thread_cnt = std::thread::hardware_concurrency();
	}
	size_t entry_per_thread = (exclusiveTo - inclusiveFrom) / thread_cnt;

	if (entry_per_thread < 1)
	{
		for (int i = inclusiveFrom; i < exclusiveTo; ++i)
		{
			func(i);
		}
		return;
	}
	std::vector<std::thread> threads;
	int start_idx, end_idx;

	for (start_idx = inclusiveFrom; start_idx < exclusiveTo; start_idx += entry_per_thread)
	{
		end_idx = start_idx + entry_per_thread;
		if (end_idx > exclusiveTo)
			end_idx = exclusiveTo;

		threads.emplace_back([&](size_t from, size_t to)
		{
			for (size_t entry_idx = from; entry_idx < to; ++entry_idx)
				func(entry_idx);
		}, start_idx, end_idx);
	}

	for (auto& t : threads)
	{
		t.join();
	}
#endif
}



void test_scale(int i, double* a, double* b) {
	a[i] = 4 * b[i];
}

int main()
{
	int N = 10000;
	double* a2 = (double*)calloc(N, sizeof(double));
	double* a1 = (double*)calloc(N, sizeof(double));
	double* b = (double*)calloc(N, sizeof(double));
	if (a1 == NULL || a2 == NULL || b == NULL)
	{
		if (a1)
		{
			free(a1);
		}if (a2)
		{
			free(a2);
		}if (b)
		{
			free(b);
		}
		return -1;
	}
	for (int i = 0; i < N; i++)
	{
		a1[i] = i;
		a2[i] = i;
		b[i] = i;
	}
	double beforeTime = bench([&] {
		for (int i = 0; i < N; i++)
		{
			test_scale(i, a1, b);
		}
	});

	std::cout << " \nbefore: " << int(beforeTime * 1000) << "ms" << std::endl;
	double afterTime = bench([&] {
		ParallelFor(0, N, [a2, b](size_t i)
		{
			test_scale(i, a2, b);
		});
	});
	std::cout << " \nafter: " << int(afterTime * 1000) << "ms" << std::endl;
 
	for (int i = 0; i < N; i++)
	{
		if (a1[i] != a2[i]) {
			printf("error %f : %f \t", a1[i], a2[i]);
			getchar();
		}
	}
	free(a1);
	free(a2);
	free(b);
	getchar();
	return 0;
}