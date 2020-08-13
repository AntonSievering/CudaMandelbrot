#pragma once

#include <cmath>
#include <iostream>
#include <string>
#include <immintrin.h>

struct Complex
{
	double r = 0.0f;
	double i = 0.0f;

	inline Complex() {}
	inline Complex(double r, double i)
	{
		this->r = r;
		this->i = i;
	}
	
	inline double cabs()
	{
		return sqrt(r * r + i * i);
	}
	inline double cabscsq()
	{
		return r * r + i * i;
	}
	inline Complex csq()
	{
		Complex cplx;
		cplx.r = r * r - i * i;
		cplx.i = 2 * r * i;
		return cplx;
	}

	std::string toString()
	{
		return std::string(std::to_string(r) + " + " + std::to_string(i) + "i");
	}

	// operator
	inline Complex operator+(Complex _other)
	{
		return Complex(r + _other.r, i + _other.i);
	}
	inline Complex operator-(Complex _other)
	{
		return Complex(r - _other.r, i - _other.i);
	}
};

size_t itFor(double r, double i, size_t maxIt, double fLimit)
{
	Complex z;
	Complex c = Complex(r, i);
	size_t nIt = 0;

	while (z.cabscsq() < fLimit && nIt < maxIt)
	{
		z = z.csq() + c;
		nIt++;
	}

	return nIt;
}

size_t* itForIntrin(double* r, double* i, size_t maxIt)
{

	__m256d _zr, _zi, _cr, _ci, _znr, _zni, _zr2, _zi2, _two, _four, _mask1;
	__m256i _it, _one, _mask2, _n;
	
	_zr = _mm256_setzero_pd(); // zr = 0
	_zi = _mm256_setzero_pd(); // zi = 0
	
	for (int j = 0; j < 4; j++)
	{
		_cr.m256d_f64[j] = r[j]; // cr = r
		_ci.m256d_f64[j] = i[j]; // ci = i
	}

	_n = _mm256_set1_epi64x(maxIt); // _n = maxIt
	_one = _mm256_set1_epi64x(1); // _one = 1
	_two = _mm256_set1_pd(2.0); // _two = 2
	_four = _mm256_set1_pd(4.0); // _four = 4
	_it = _mm256_setzero_si256(); // it = 0

Repeat:
	// Step 1: z = z * z + c
	_zr2 = _mm256_mul_pd(_zr, _zr); // zr2 = zr * zr
	_zi2 = _mm256_mul_pd(_zi, _zi); // zi2 = zi * zi
	_znr = _mm256_add_pd(_mm256_sub_pd(_zr2, _zi2), _cr); // znr = zr2 - zi2 + cr
	_zni = _mm256_add_pd(_mm256_mul_pd(_mm256_mul_pd(_two, _zr), _zi), _ci); // zni = 2 * zr * zi + ci
	_zr = _znr; // zr = znr
	_zi = _zni; // zi = zni

	// Step 2: goto Repeat if nessesary
	_znr = _mm256_add_pd(_zr2, _zi2); // znr = zr2 + zi2
	_mask1 = _mm256_cmp_pd(_znr, _four, _CMP_LT_OQ); // mask1 = _znr < _four
	_mask2 = _mm256_cmpgt_epi64(_it, _n); // mask2 = _it > _n
	_mask2 = _mm256_and_si256(_mask2, _mm256_castpd_si256(_mask1)); // _mask2 = _mask2 && _mask1
	
	_it = _mm256_add_epi64(_it, _mm256_and_si256(_one, _mask2)); // it += (1 & _mask2)

	if (_mm256_movemask_pd(_mm256_castsi256_pd(_mask2)) > 0) goto Repeat;

	size_t* pResult = new size_t[4];
	for (int i = 0; i < 4; i++) pResult[i] = _it.m256i_i64[3 - i];

	return pResult;
}