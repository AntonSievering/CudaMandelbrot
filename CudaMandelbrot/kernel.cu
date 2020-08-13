#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "MandelBrot.h"
#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"
#include <thread>
#include <fstream>

#define CPI 0.15915494309189

__device__ double cudaSin(double r)
{
	r *= CPI;
	double v = -16 * r * r + 8 * r;
	if (r < 0.5) return v;
	return -v;
}

// z0 = c
// zn+1 = zn * zn + c
__global__ void Mandelbrot(const double fBeginX, const double fBeginY, const double fIncrease, size_t* pIterations, size_t nMaxIterations, double fLimit, int nElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int x = i % 1920;
	int y = i / 1920;

	if (i < nElements)
	{
		double cr, ci, zr, zi, znr, zni;
		size_t nIterations = 0;

		cr = fBeginX + (double)x * fIncrease;
		ci = fBeginY + (double)y * fIncrease;
		zr = cr;
		zi = ci;

		while (zr * zr + zi * zi < fLimit && nIterations < nMaxIterations)
		{
			znr = zr * zr - zi * zi + cr;
			zni = 2 * zr * zi + ci;
			zr = znr;
			zi = zni;

			nIterations++;
		}
		pIterations[i] = nIterations;
	}
}

// z0 = c
// zn+1 = zn * zn - c
__global__ void Fractal0(const double fBeginX, const double fBeginY, const double fIncrease, size_t* pIterations, size_t nMaxIterations, double fLimit, int nElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int x = i % 1920;
	int y = i / 1920;

	if (i < nElements)
	{
		double cr, ci, zr, zi, znr, zni;
		size_t nIterations = 0;

		cr = fBeginX + (double)x * fIncrease;
		ci = fBeginY + (double)y * fIncrease;
		zr = cr;
		zi = ci;

		while (zr * zr + zi * zi < fLimit && nIterations < nMaxIterations)
		{
			znr = zr * zr - zi * zi - cr;
			zni = 2 * zr * zi - ci;
			zr = znr;
			zi = zni;

			nIterations++;
		}
		pIterations[i] = nIterations;
	}
}

// z0 = c
// zn+1 = zn * zn + 2 * c
__global__ void Fractal1(const double fBeginX, const double fBeginY, const double fIncrease, size_t* pIterations, size_t nMaxIterations, double fLimit, int nElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int x = i % 1920;
	int y = i / 1920;

	if (i < nElements)
	{
		double cr, ci, zr, zi, znr, zni;
		size_t nIterations = 0;

		cr = fBeginX + (double)x * fIncrease;
		ci = fBeginY + (double)y * fIncrease;
		zr = cr;
		zi = ci;

		while (zr * zr + zi * zi < fLimit && nIterations < nMaxIterations)
		{
			znr = zr * zr - zi * zi + 2 * cr;
			zni = 2 * zr * zi + 2 * ci;
			zr = znr;
			zi = zni;

			nIterations++;
		}
		pIterations[i] = nIterations;
	}
}

// z0 = c
// zn+1 = zn * zn * zn + c
__global__ void Fractal2(const double fBeginX, const double fBeginY, const double fIncrease, size_t* pIterations, size_t nMaxIterations, double fLimit, int nElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int x = i % 1920;
	int y = i / 1920;

	if (i < nElements)
	{
		double cr, ci, zr, zi, znr, zni;
		size_t nIterations = 0;

		cr = fBeginX + (double)x * fIncrease;
		ci = fBeginY + (double)y * fIncrease;
		zr = cr;
		zi = ci;

		while (zr * zr + zi * zi < fLimit && nIterations < nMaxIterations)
		{
			znr = zr * zr * zr - 3 * zr * zi * zi;
			zni = 3 * zr * zr * zi - zi * zi * zi;
			zr = znr + cr;
			zi = zni + ci;

			nIterations++;
		}
		pIterations[i] = nIterations;
	}
}

// z0 = c
// zn+1 = zn * zn * zn * zn + c
__global__ void Fractal3(const double fBeginX, const double fBeginY, const double fIncrease, size_t* pIterations, size_t nMaxIterations, double fLimit, int nElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int x = i % 1920;
	int y = i / 1920;

	if (i < nElements)
	{
		double cr, ci, zr, zi, znr, zni;
		size_t nIterations = 0;

		cr = fBeginX + (double)x * fIncrease;
		ci = fBeginY + (double)y * fIncrease;
		zr = cr;
		zi = ci;

		while (zr * zr + zi * zi < fLimit && nIterations < nMaxIterations)
		{
			znr = zr * zr * zr * zr - 6 * zr * zr * zi * zi + zi * zi * zi * zi;
			zni = 4 * zr * zr * zr * zi - 4 * zr * zi * zi * zi;
			zr = znr + cr;
			zi = zni + ci;

			nIterations++;
		}
		pIterations[i] = nIterations;
	}
}

// z0 = c
// zn+1 = zn * zn * zn * zn + c
__global__ void Fractal4(const double fBeginX, const double fBeginY, const double fIncrease, size_t* pIterations, size_t nMaxIterations, double fLimit, int nElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int x = i % 1920;
	int y = i / 1920;

	if (i < nElements)
	{
		double cr, ci, zr, zi, znr, zni;
		size_t nIterations = 0;

		cr = fBeginX + (double)x * fIncrease;
		ci = fBeginY + (double)y * fIncrease;
		zr = cr;
		zi = ci;

		while (zr * zr + zi * zi < fLimit && nIterations < nMaxIterations)
		{
			znr = zr * zr - zi * zi + zr + cr;
			zni = 2 * zr * zi + zi + ci;
			zr = znr + cr;
			zi = zni + ci;

			nIterations++;
		}
		pIterations[i] = nIterations;
	}
}

using namespace std::chrono_literals;

class MandelBrot : public olc::PixelGameEngine
{
private:
	std::vector<olc::Key> vKeys;
	size_t nFractal = 0;

	size_t nMaxIterations = 255;
	olc::Pixel* pPallette = nullptr;
	double fLimit = 1024.0;

	olc::Sprite* sprMandelbrot = nullptr;
	olc::Decal* decMandelbrot = nullptr;
	olc::Sprite* sprXAxis = nullptr;
	olc::Sprite* sprYAxis = nullptr;
	olc::Decal* decXAxis = nullptr;
	olc::Decal* decYAxis = nullptr;

	size_t nThreads = 1;
	std::thread* vThreadPool;
	std::atomic<bool>* bThreadDone;
	bool bRenderStarted = false;
	bool bRendererRestartRequest = false;
	bool bProgrammRunning = true;

	bool bShowAxis = true;
	bool bShowCoords = false;
	bool bSelectionBlocked = false;
	bool bRecording = false;
	size_t nFramesDone = 0;
	std::string sFramesFolderName = "frames";
	size_t nColorMode = 1;

	// pan and zoom
	olc::vd2d panOffset = { 9.44, 5.37 };
	olc::vd2d panStart = olc::vd2d();
	double fZoom = 100.0;

	olc::vi2d vSelectedStart = { -1, -1 };
	olc::vi2d vSelectedSize = { 0, 0 };
	olc::Sprite* sprGrid = nullptr;
	olc::Decal* decGrid = nullptr;

public:
	MandelBrot()
	{
		sAppName = "MandelBrot";
	}

public:
	olc::vd2d WorldToScreen(const olc::vd2d vWorld)
	{
		return (vWorld + panOffset) * fZoom;
	}
	olc::vd2d ScreenToWorld(const olc::vd2d vScreen)
	{
		return vScreen / fZoom - panOffset;
	}

	olc::Pixel GenerateColor(size_t nIterations)
	{
		switch (nColorMode)
		{
		case  1: return GenerateColor1(nIterations); break;
		default: return GenerateColor2(nIterations); break;
		}
	}
	olc::Pixel GenerateColor1(size_t nIterations)
	{
		float a = 0.1f;
		return olc::PixelF(0.5f * sin(a * nIterations) + 0.5f, 0.5f * sin(a * nIterations + 2.094f) + 0.5f, 0.5f * sin(a * nIterations + 4.188f) + 0.5f);
	}
	olc::Pixel GenerateColor2(size_t nIterations)
	{
		int a = 255 - 255 * (double)nIterations / nMaxIterations;
		return olc::Pixel(a, a, a, 255);
	}

	void GeneratePallette()
	{
		olc::Pixel* pPalletteOld = pPallette;
		pPallette = new olc::Pixel[nMaxIterations + 1];

		for (int i = 0; i < nMaxIterations + 1; i++)
		{
			pPallette[i] = GenerateColor(i);
		}

		delete[] pPalletteOld;
	}
	void ClearPixel()
	{
		for (int x = 0; x < 1920; x++)
		{
			for (int y = 0; y < 1080; y++)
			{
				sprMandelbrot->SetPixel(x, y, olc::BLACK);
			}
		}
	}
	void MandelbrotThreadCPU(size_t id, size_t fromX, size_t toX)
	{
		while (true)
		{
			while (bThreadDone[id])
			{
				std::this_thread::sleep_for(1ms);
			}

			for (int x = fromX; x < toX; x++)
			{
				for (int y = 0; y < 1080; y++)
				{
					olc::vd2d world = ScreenToWorld(olc::vi2d(x, y));
					size_t it = itFor(world.x, world.y, nMaxIterations, fLimit);
					sprMandelbrot->SetPixel(x, y, GenerateColor(it));
				}
			}
			bThreadDone[id] = true;
		}
	}
	void MandelbrotThreadCPUIntrinsic(size_t id, size_t fromX, size_t toX)
	{
		while (true)
		{
			while (bThreadDone[id])
			{
				std::this_thread::sleep_for(1ms);
			}

			for (int x = fromX; x < toX; x++)
			{
				for (int y = 0; y < 1080; y += 4)
				{
					double* r = new double[4]; double* i = new double[4];
					for (int j = 0; j < 4; j++)
					{
						auto vWorld = ScreenToWorld(olc::vi2d(x, y + j));
						r[j] = vWorld.x;
						i[j] = vWorld.y;
					}
					
					size_t* it = itForIntrin(r, i, nMaxIterations);

					for (int j = 0; j < 4; j++) sprMandelbrot->SetPixel(x, y, GenerateColor(it[j]));
				}
			}
			bThreadDone[id] = true;
		}
	}
	void MandelbrotThreadCuda(size_t id, size_t fromX, size_t toX)
	{
		GeneratePallette();

		size_t nSize = 1920 * 1080;
		size_t nThreads = 256;
		size_t nBlocks = nSize / nThreads;

		// host memory
		size_t* h_pIterations = (size_t*)malloc(nSize * sizeof(size_t));

		// Device Memory
		size_t* d_pIterations = nullptr;
		// Allocate Memory
		cudaMalloc((void**)&d_pIterations, nSize * sizeof(size_t));

		while (bProgrammRunning)
		{
			while (bThreadDone[id])
			{
				std::this_thread::sleep_for(100us);
			}

			olc::vd2d vWorldStart = ScreenToWorld({0.0, 0.0});
			double fWorldInc = ScreenToWorld({ 1, 0 }).x - vWorldStart.x;

			// Start the kernel
			switch (nFractal)
			{
			case 0: Mandelbrot <<< nBlocks, nThreads >>> (vWorldStart.x, vWorldStart.y, fWorldInc, d_pIterations, nMaxIterations, fLimit, nSize); break;
			case 1: Fractal0 <<< nBlocks, nThreads >>> (vWorldStart.x, vWorldStart.y, fWorldInc, d_pIterations, nMaxIterations, fLimit, nSize);   break;
			case 2: Fractal1 <<< nBlocks, nThreads >>> (vWorldStart.x, vWorldStart.y, fWorldInc, d_pIterations, nMaxIterations, fLimit, nSize);   break;
			case 3: Fractal2 <<< nBlocks, nThreads >>> (vWorldStart.x, vWorldStart.y, fWorldInc, d_pIterations, nMaxIterations, fLimit, nSize);   break;
			case 4: Fractal3 <<< nBlocks, nThreads >>> (vWorldStart.x, vWorldStart.y, fWorldInc, d_pIterations, nMaxIterations, fLimit, nSize);   break;
			case 5: Fractal4 <<< nBlocks, nThreads >>> (vWorldStart.x, vWorldStart.y, fWorldInc, d_pIterations, nMaxIterations, fLimit, nSize);   break;
			}
			
			cudaMemcpy(h_pIterations, d_pIterations, nSize * sizeof(size_t), cudaMemcpyDeviceToHost);
			
			for (size_t j = 0; j < nSize; j++)
			{
				size_t x = j % 1920;
				size_t y = j / 1920;
				sprMandelbrot->SetPixel(olc::vi2d(x, y), pPallette[h_pIterations[j]]);
			}

			bThreadDone[id] = true;

			// if in recording mode, save
			if (bRecording)
			{
				auto GenString = [&](int n, int digits)
				{
					std::string str = std::string();
					std::string strn = std::to_string(n);
					int nlen = strn.length();
					for (int i = 0; i < digits - nlen; i++) str.append("0");
					return str + strn;
				};

				std::string sFileName = sFramesFolderName + "/frame" + GenString(nFramesDone, 5) + ".frame";
				std::ofstream file = std::ofstream(sFileName);
				if (file.is_open())
				{
					for (size_t y = 0; y < 1080; y++)
					{
						for (size_t x = 0; x < 1920; x++)
						{
							int idx = y * 1920 + x;
							file << h_pIterations[idx];
							if (x == 1919 && y != 1079) file << "\n";
							else if (x != 1919) file << ",";
						}
					}

					file.close();
				}

				nFramesDone++;
			}
		}

		// free the memory
		free(h_pIterations);
		cudaFree(d_pIterations);
	}
	void StartThreads()
	{
		for (int i = 0; i < nThreads; i++)
		{
			bThreadDone[i] = false;
		}
	}
	bool ThreadsDone()
	{
		bool bThreadsDone = true;
		for (int i = 0; i < nThreads; i++)
		{
			bThreadsDone *= bThreadDone[i];
		}
		return bThreadsDone;
	}
	void WaitForThreads()
	{
		while (!ThreadsDone())
		{
			std::this_thread::sleep_for(1ms);
		}
	}

	bool OnUserCreate() override
	{
		sprMandelbrot = new olc::Sprite(1920, 1080);
		decMandelbrot = new olc::Decal(sprMandelbrot);

		vThreadPool = new std::thread[nThreads];
		bThreadDone = new std::atomic<bool>[nThreads];
		size_t pxPerThread = 1920 / nThreads;
		for (int i = 0; i < nThreads; i++)
		{
			vThreadPool[i] = std::thread(&MandelBrot::MandelbrotThreadCuda, this, i, i * pxPerThread, (i + 1) * pxPerThread);
			bThreadDone[i] = false;
		}

		sprXAxis = new olc::Sprite(1920, 1);
		for (int i = 0; i < 1920; i++) sprXAxis->SetPixel(olc::vi2d(i, 0), olc::BLACK);
		decXAxis = new olc::Decal(sprXAxis);
		sprYAxis = new olc::Sprite(1, 1080);
		for (int i = 0; i < 1080; i++) sprYAxis->SetPixel(olc::vi2d(0, i), olc::BLACK);
		decYAxis = new olc::Decal(sprYAxis);

		sprGrid = new olc::Sprite(1920, 1080);
		for (int x = 0; x < 1920; x++)
		{
			for (int y = 0; y < 1080; y++)
			{
				sprGrid->SetPixel(x, y, olc::BLANK);
			}
		}
		decGrid = new olc::Decal(sprGrid);

		vKeys.push_back(olc::Key::K1);
		vKeys.push_back(olc::Key::K2);
		vKeys.push_back(olc::Key::K3);
		vKeys.push_back(olc::Key::K4);
		vKeys.push_back(olc::Key::K5);
		vKeys.push_back(olc::Key::K6);

		return true;
	}
	bool OnUserUpdate(float fElapsedTime) override
	{
		// Draw Hint
		olc::Pixel col;
		olc::vf2d vfScale = { 2.0, 2.0 };
		switch (nColorMode)
		{
		case 0: col = olc::WHITE;
		case 1: col = olc::BLACK;
		}

		auto DrawSpriteLine = [&](olc::Sprite* spr, olc::Pixel col, olc::vi2d pos, int length, int strength, bool bX = true)
		{
			if (length < 0)
			{
				if (bX) pos.x += length;
				else pos.y += length;
				length = -length;
			}

			if (bX)
			{
				for (int x = pos.x; x < pos.x + length; x++)
				{
					for (int y = pos.y; y < pos.y + strength; y++)
					{
						spr->SetPixel({ x, y }, col);
					}
				}
			}
			else
			{
				for (int y = pos.y; y < pos.y + length; y++)
				{
					for (int x = pos.x; x < pos.x + strength; x++)
					{
						spr->SetPixel({ x, y }, col);
					}
				}
			}
		};
		auto DrawSpriteRect = [&](olc::Sprite* spr, olc::Pixel col, olc::vi2d pos, olc::vi2d size, int strength)
		{
			DrawSpriteLine(spr, col, pos, size.x, strength, true);
			DrawSpriteLine(spr, col, pos, size.y, strength, false);
			DrawSpriteLine(spr, col, pos + olc::vi2d(0, size.y), size.x, strength, true);
			DrawSpriteLine(spr, col, pos + olc::vi2d(size.x, 0), size.y, strength, false);
		};

		int i = 0;
		for (auto& key : vKeys)
		{
			if (GetKey(key).bPressed)
			{
				nFractal = i;
				bRendererRestartRequest = true;
			}
			i++;
		}

		Clear(olc::BLACK);
		olc::vi2d mouse = { GetMouseX(), GetMouseY() };

		if (!bRenderStarted)
		{
			StartThreads();
			bRenderStarted = true;
		}

		// Draw the Mandelbrot
		delete decMandelbrot;
		decMandelbrot = new olc::Decal(sprMandelbrot);
		if (ThreadsDone())
		{
			if (bRendererRestartRequest)
			{
				bRendererRestartRequest = false;
				bRenderStarted = false;
			}
		}

		DrawDecal({ 0, 0 }, decMandelbrot);

		if (GetMouse(1).bPressed) panStart = mouse;
		if (GetMouse(1).bHeld)
		{
			panOffset += olc::vd2d((olc::vd2d)(mouse - panStart) / fZoom);
			panStart = mouse;
			bRendererRestartRequest = true;
		}

		if (GetKey(olc::Key::SPACE).bPressed) bShowAxis = !bShowAxis;
		if (bShowAxis)
		{
			olc::vd2d vWorldZero = WorldToScreen({ 0.0, 0.0 });
			DrawDecal(olc::vi2d(vWorldZero.x, 0), decYAxis);
			DrawDecal(olc::vi2d(0, vWorldZero.y), decXAxis);
			DrawStringDecal(olc::vi2d(vWorldZero.x + 10, 10), "Imag [C]", col, { 1.0, 1.0 });
			DrawStringDecal(olc::vi2d(ScreenWidth() - 74, vWorldZero.y - 18), "Real [C]", col, { 1.0, 1.0 });
		}

		// Edit iterations and limits
		if (GetKey(olc::Key::U).bPressed && nMaxIterations > 1)
		{
			nMaxIterations /= 2;
			GeneratePallette();
			bRendererRestartRequest = true;
		}
		if (GetKey(olc::Key::I).bPressed)
		{
			nMaxIterations *= 2;
			GeneratePallette();
			bRendererRestartRequest = true;
		}
		if (GetKey(olc::Key::O).bPressed && nMaxIterations > 1)
		{
			nMaxIterations -= 1;
			GeneratePallette();
			bRendererRestartRequest = true;
		}
		if (GetKey(olc::Key::P).bPressed)
		{
			nMaxIterations += 1;
			GeneratePallette();
			bRendererRestartRequest = true;
		}
		if (GetKey(olc::Key::T).bPressed)
		{
			fLimit /= 2.0;
			bRendererRestartRequest = true;
		}
		if (GetKey(olc::Key::Z).bPressed)
		{
			fLimit *= 2.0;
			bRendererRestartRequest = true;
		}
		if (GetKey(olc::Key::ENTER).bPressed)
		{
			fZoom = 100;
			panOffset = { 9.44, 5.37 };
			panStart = { 0, 0 };
			bRendererRestartRequest = true;
		}

		// Recording
		if (GetKey(olc::Key::F1).bPressed)
		{
			bRecording = !bRecording;
			if (bRecording) bRenderStarted = false;
		}

		// Zoom and pan stuff
		olc::vd2d vMouseBeforeZoom = ScreenToWorld(mouse);
		if (GetKey(olc::Key::Q).bPressed || GetKey(olc::Key::Q).bHeld) fZoom += fZoom * 1.1 * fElapsedTime;
		if (bRecording && !bRenderStarted) fZoom += fZoom * 1.1 / 120.0;
		if (GetKey(olc::Key::E).bPressed || GetKey(olc::Key::E).bHeld) fZoom -= fZoom * 1.1 * fElapsedTime;
		olc::vd2d vMouseAfterZoom = ScreenToWorld(mouse);
		if ((vMouseAfterZoom - vMouseBeforeZoom) != olc::vd2d())
		{
			panOffset += (vMouseAfterZoom - vMouseBeforeZoom);
			bRendererRestartRequest = true;
		}

		// halt if ESC is pressed
		if (GetKey(olc::Key::ESCAPE).bPressed && !GetMouse(0).bHeld) return false;

		// Set Color Mode
		if (GetKey(olc::Key::J).bPressed)
		{
			nColorMode = 0;
			GeneratePallette();
			bRendererRestartRequest = true;
		}
		if (GetKey(olc::Key::K).bPressed)
		{
			nColorMode = 1;
			GeneratePallette();
			bRendererRestartRequest = true;
		}

		if (GetKey(olc::Key::TAB).bPressed) bShowCoords = !bShowCoords;
		if (bShowCoords)
		{
			olc::vd2d vMouseWorld = ScreenToWorld(mouse + olc::vi2d(50, 0));
			olc::Pixel col = olc::BLACK; if (sprMandelbrot->GetPixel(mouse) == olc::BLACK) col = olc::WHITE;
			std::string sCoord = std::to_string(vMouseWorld.x) + " + " + std::to_string(-vMouseWorld.y) + "i";
			DrawStringDecal(mouse, sCoord, col, { 2.0, 2.0 });
		}

		// Draw Selected grid
		if (GetMouse(0).bPressed && !bSelectionBlocked) vSelectedStart = mouse;
		if (GetMouse(0).bHeld && !bSelectionBlocked)
		{
			// Vanish the old grid
			DrawSpriteRect(sprGrid, olc::BLANK, vSelectedStart, vSelectedSize, 2);
			
			// Set the new grid
			vSelectedSize = mouse - vSelectedStart;

			// Draw Grid to the Sprite
			DrawSpriteRect(sprGrid, olc::RED, vSelectedStart, vSelectedSize, 2);

			// Draw Grid to Screen
			delete decGrid;
			decGrid = new olc::Decal(sprGrid);
			SetPixelMode(olc::Pixel::ALPHA);
			DrawDecal({ 0, 0 }, decGrid);
			SetPixelMode(olc::Pixel::NORMAL);
		}
		if (GetMouse(0).bReleased && !bSelectionBlocked)
		{
			DrawSpriteRect(sprGrid, olc::BLANK, vSelectedStart, vSelectedSize, 2);

			if (vSelectedSize.x != 0 && vSelectedSize.y != 0)
			{
				if (vSelectedSize.x < 0)
				{
					vSelectedSize.x = -vSelectedSize.x;
					vSelectedStart.x -= vSelectedSize.x;
				}
				if (vSelectedSize.y < 0)
				{
					vSelectedSize.y = -vSelectedSize.y;
					vSelectedStart.y -= vSelectedSize.y;
				}

				// Zoom in
				panOffset -= (olc::vd2d)(vSelectedStart) / fZoom;
				fZoom *= 1920.0 / (double)vSelectedSize.x;

				bRendererRestartRequest = true;
			}
		}
		if (!GetMouse(0).bHeld && !GetMouse(0).bPressed && !GetMouse(0).bReleased) bSelectionBlocked = false;
		else
		{
			if (GetKey(olc::Key::ESCAPE).bPressed)
			{
				bSelectionBlocked = true;
				DrawSpriteRect(sprGrid, olc::BLANK, vSelectedStart, vSelectedSize, 2);
			}
		}

		DrawStringDecal({ 5, 25 }, "Bedienungshilfe:", col, vfScale);
		DrawStringDecal({ 5, 45 }, "<J> Helligkeitsstufen", col, vfScale);
		DrawStringDecal({ 5, 65 }, "<K> Bunte Farben", col, vfScale);
		DrawStringDecal({ 5, 85 }, "<ESC> Beenden", col, vfScale);
		DrawStringDecal({ 5, 105 }, "<Q> Hineinzoomen", col, vfScale);
		DrawStringDecal({ 5, 125 }, "<E> Hinauszoomen", col, vfScale);
		std::string s = bShowAxis ? "verstecken" : "zeigen";
		DrawStringDecal({ 5, 145 }, "<SPACE> Koordinatensystem " + s, col, vfScale);
		s = bShowCoords ? "verstecken" : "zeigen";
		DrawStringDecal({ 5, 165 }, "<TAB> Korrdinate " + s, col, vfScale);
		DrawStringDecal({ 5, 185 }, "<U> maximale Interationen halbieren", col, vfScale);
		DrawStringDecal({ 5, 205 }, "<I> maximale Iterationen verdoppeln", col, vfScale);
		DrawStringDecal({ 5, 225 }, "<O> maximale Iterationen um 1 inkrementieren", col, vfScale);
		DrawStringDecal({ 5, 245 }, "<P> maximale Iterationen um 1 dekrementieren", col, vfScale);
		DrawStringDecal({ 5, 265 }, "<T> Limit halbieren", col, vfScale);
		DrawStringDecal({ 5, 285 }, "<Z> Limit verdoppeln", col, vfScale);
		DrawStringDecal({ 5, 305 }, "<ENTER> Standart-Zoom", col, vfScale);

		DrawStringDecal({ 5, 345 }, "aktuelles Limit:" + std::to_string(fLimit), col, vfScale);
		DrawStringDecal({ 5, 365 }, "aktuelle maximale Iterationen: " + std::to_string(nMaxIterations), col, vfScale);
		DrawStringDecal({ 5, 385 }, "Zoom: " + std::to_string(fZoom), col, vfScale);
		DrawStringDecal({ 5, 405 }, "Du nimmst gerade " + (std::string)(bRecording ? "" : "nicht ") + "auf", col, vfScale);

		DrawStringDecal({ 1500, 25 }, "aufgenommene Frames: " + std::to_string(nFramesDone), col, vfScale);

		return true;
	}
	bool OnUserDestroy() override
	{
		bProgrammRunning = true;
		for (int i = 0; i < nThreads; i++)
		{
			vThreadPool[i].detach();
		}
		return true;
	}
};

int main()
{
	MandelBrot brot;
	if (brot.Construct(1920, 1080, 1, 1, true))
		brot.Start();
	return 0;
}