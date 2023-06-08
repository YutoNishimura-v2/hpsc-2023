// Required: module load cuda/12.1.0

#include <cstdio>
#include <cmath>
#include <vector>

__global__ void cavity(
    float *u, float *v, float *b, float *p, float *pn, float *un, float *vn, int nx, int ny, int nt,int nit, int size, float dx, float dy, float dt, int rho, float nu
){

    for(int n = 0; n <= nt; n++){
		for(int j = 1; j <= ny - 2; j++){
			for(int i = 1; i <= nx - 2; i++){
				b[j*nx+i] = rho * (1.0 / dt *\
						((u[j*nx+i+1] - u[j*nx+i-1]) / (2.0 * dx) + (v[(j+1)*nx+i] - v[(j-1)*nx+i]) / (2.0 * dy)) -\
						pow((u[j*nx+i+1] - u[j*nx+i-1]) / (2.0 * dx),2) - 2.0 * ((u[(j+1)*nx+i] - u[(j-1)*nx+i]) / (2.0 * dy) *\
						(v[j*nx+i+1] - v[j*nx+i-1]) / (2.0 * dx)) - pow((v[(j+1)*nx+i] - v[(j-1)*nx+i]) / (2.0 * dy),2.0));
			}	
		}
		for(int it = 0; it <= nit; it++){
			for(int j = 0; j <= size - 1; j++){    		
				pn[j] = p[j];	
			}
			for(int j = 1; j <= ny - 2 ; j++){
				for(int i = 1; i <= nx - 2; i++){
					p[j*nx+i] = (pow(dy,2)*(pn[j*nx+i+1] + pn[j*nx+i-1]) +\
							pow(dx,2)*(pn[(j+1)*nx+i] + pn[(j-1)*nx+i]) -\
							b[j*nx+i] * pow(dx,2) * pow(dy,2))\
							/ (2 * (pow(dx,2) + pow(dy,2)));
				}
			}
			for(int i = 0; i <= ny - 1; i++){
		       	p[i*nx+ny-1] = p[i*nx+ny-2];
			}
			for(int i = 0; i <= nx - 1; i++){
                p[i] = p[nx+i];
            }
			for(int i = 0; i <= ny - 1; i++){
                p[i*nx] = p[i*nx+1];
            }
			for(int i = 0; i <= ny - 1; i++){
                p[(nx-1)*nx+i] = 0;
            }
		}
        for(int j = 0; j <= size - 1; j++){    		
				un[j] = u[j];	
			}

        for(int j = 0; j <= size - 1; j++){    		
				vn[j] = v[j];	
			}
		for(int j = 1; j <= ny - 2; j++){
			for(int i = 1; i <= nx - 2; i++){
				u[j*nx+i] = un[j*nx+i] -un[j*nx+i] * dt / dx *(un[j*nx+i] - un[j*nx+i-1])\
					  - un[j*nx+i] * dt / dy * (un[j*nx+i] - un[(j-1)*nx+i])\
					  - dt / (2 * rho * dx) * (p[j*nx+i+1] - p[j*nx+i-1])\
					  + nu * dt / pow(dx,2) * (un[j*nx+i+1] - 2 * un[j*nx+i] +un[j*nx+i-1])\
					  + nu * dt / pow(dy,2) * (un[(j+1)*nx+i] -2 * un[j*nx+i] + un[(j-1)*nx+i]);

				v[j*nx+i] = vn[j*nx+i] -vn[j*nx+i] * dt / dx *(vn[j*nx+i] - vn[j*nx+i-1])\
                                          - vn[j*nx+i] * dt / dy * (vn[j*nx+i] - vn[(j-1)*nx+i])\
                                          - dt / (2 * rho * dx) * (p[(j+1)*nx+i] - p[(j-1)*nx+i])\
                                          + nu * dt / pow(dx,2) * (vn[j*nx+i+1] - 2 * vn[j*nx+i] +vn[j*nx+i-1])\
                                          + nu * dt / pow(dy,2) * (vn[(j+1)*nx+i] -2 * vn[j*nx+i] + vn[(j-1)*nx+i]);
			}
		}
		for(int i = 0; i <= nx - 1; i++){
			u[i] = 0;
		}
		for(int i = 0; i <= nx - 1; i++){
            u[i*nx] = 0;
        } 
		for(int i = 0; i <= nx - 1; i++){
            u[i*nx+nx-1] = 0;
        }
		for(int i = 0; i <= nx - 1; i++){
            u[(nx-1)*nx+i] = 1;
        }
		for(int i = 0; i <= nx - 1; i++){
            v[i] = 0;
        }
		for(int i = 0; i <= nx - 1; i++){
            v[(nx-1)*nx+i] = 0;
        }
		for(int i = 0; i <= nx - 1; i++){
            v[i*nx] = 0;
        }
		for(int i = 0; i <= nx - 1; i++){
            v[i*nx+nx-1] = 0;
        }
	}
}

int main(){
	int nx = 41;
	int ny = 41;
	int nt = 500;
	int nit = 50;
    int size = nx * ny;
	float dx = 2.0/(nx - 1);
	float dy = 2.0/(ny - 1);
	float dt = 0.01;
	int rho = 1;
	float nu = 0.02;

    const int n = size;
    const int M = 64;

    std::vector<float> u(size,0.0);
    std::vector<float> v(size,0.0);
    std::vector<float> b(size,0.0);
    std::vector<float> p(size,0.0);
    std::vector<float> pn(size,0.0);
    std::vector<float> un(size,0.0);
    std::vector<float> vn(size,0.0);

    float* du;
    float* dv;
    float* db;
    float* dp;
    float* dpn;
    float* dun;
    float* dvn;

    cudaMallocManaged(&du, size*sizeof(float));
    cudaMallocManaged(&dv, size*sizeof(float));
    cudaMallocManaged(&db, size*sizeof(float));
    cudaMallocManaged(&dp, size*sizeof(float));
    cudaMallocManaged(&dpn, size*sizeof(float));
    cudaMallocManaged(&dun, size*sizeof(float));
    cudaMallocManaged(&dvn, size*sizeof(float));
    void *args[] = {
        (void *)&du,  (void *)&dv, (void *)&db, (void *)&dp,
        (void *)&dpn, (void *)&dun, (void *)&dvn,
        (void *)&nx, (void *)&ny, (void *)&nt, (void *)&nit,
        (void *)&size, (void *)&dx, (void *)&dy, (void *)&dt,
        (void *)&rho, (void *)&nu,
    };
    cudaLaunchCooperativeKernel(
        (void*)cavity, (n+M-1)/M, M, args
    );
    cudaDeviceSynchronize();
    cudaFree(du);
    cudaFree(dv);
    cudaFree(db);
    cudaFree(dp);
    cudaFree(dpn);
    cudaFree(dun);
    cudaFree(dvn);
}