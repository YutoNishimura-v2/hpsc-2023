// Required: module load gcc

#include<iostream>
#include<vector>
#include <matplotlibcpp.h>

namespace plt = matplotlibcpp;

using namespace std;
using matrix = vector<vector<float>>;

const int nx = 41;
const int ny = 41;
const int nt = 500;
const int nit = 50;
const float dx = 2. / (nx - 1);
const float dy = 2. / (ny - 1);
const float dt = .01;
const float rho = 1;
const float nu = .02;

int main() {
    vector<float> x(nx);
    vector<float> y(ny);

    for (int i = 0; i < nx; i++) {
        x[i] = i * dx;
    }

    for (int j = 0; j < ny; j++) {
        y[j] = j * dy;
    }

    matrix u(ny, vector<float>(nx, 0));
    matrix v(ny, vector<float>(nx, 0));
    matrix p(ny, vector<float>(nx, 0));
    matrix b(ny, vector<float>(nx, 0));
    matrix un(ny, vector<float>(nx, 0));
    matrix vn(ny, vector<float>(nx, 0));
    matrix pn(ny, vector<float>(nx, 0));

    for (int n = 0; n < nt; n++) {
        for (int j = 1; j < ny-1; j++) {
            for (int i = 1; i < nx - 1; i++) {
             b[j][i] = rho * (1 / dt *
                     ((u[j][i+1] - u[j][i-1]) / (2 * dx) + (v[j+1][i] - v[j-1][i]) / (2 * dy)) -
                     ((u[j][i+1] - u[j][i-1]) / (2 * dx)) * ((u[j][i+1] - u[j][i-1]) / (2 * dx)) - 2 * ((u[j+1][i] - u[j-1][i]) / (2 * dy) *
                      (v[j][i+1] - v[j][i-1]) / (2 * dx)) - ((v[j+1][i] - v[j-1][i]) / (2 * dy)) * ((v[j+1][i] - v[j-1][i]) / (2 * dy)));
            }
        }

        for (int it = 0; it < nit; it++) {
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    pn[j][i] = p[j][i];
                }
            }

            for (int j = 1; j < ny - 1; j++) {
                for (int i = 1; i < nx - 1; i++) {
                    p[j][i] = (dy * dy * (pn[j][i+1] + pn[j][i-1]) +
                               dx * dx * (pn[j+1][i] + pn[j-1][i]) -
                               b[j][i] * dx * dx * dy * dy)\
                              / (2 * (dx * dx + dy * dy));
                }
            }
            for (int j = 1; j < ny - 1; j++) {
                p[j][nx - 1] = p[j][nx-2];
                p[j][0] = p[j][1];
            }

            for (int i = 1; i < nx - 1; i++) {
                p[0][i] = p[1][i];
                p[ny - 1][i] = 0;
            }
        }
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                un[j][i] = u[j][i];
                vn[j][i] = v[j][i];
            }
        }

        for (int j = 1; j < ny - 1; j++) {
            for (int i = 1; i < nx - 1; i++) {
                u[j][i] = un[j][i] - un[j][i] * dt / dx * (un[j][i] - un[j][i - 1])
                                   - un[j][i] * dt / dy * (un[j][i] - un[j - 1][i])
                                   - dt / (2 * rho * dx) * (p[j][i+1] - p[j][i-1])\
                                   + nu * dt / dx * dx * (un[j][i+1] - 2 * un[j][i] + un[j][i-1])
                                   + nu * dt / dy * dy * (un[j+1][i] - 2 * un[j][i] + un[j-1][i]);

                v[j][i] = vn[j][i] - vn[j][i] * dt / dx * (vn[j][i] - vn[j][i - 1])
                                   - vn[j][i] * dt / dy * (vn[j][i] - vn[j - 1][i])
                                   - dt / (2 * rho * dx) * (p[j+1][i] - p[j-1][i])
                                   + nu * dt / dx * dx * (vn[j][i+1] - 2 * vn[j][i] + vn[j][i-1])
                                   + nu * dt / dy * dy * (vn[j+1][i] - 2 * vn[j][i] + vn[j-1][i]);
            }
        }

        for (int j = 0; j < ny; j++) {
            u[j][0]  = 0;
            u[j][nx - 1] = 0;
            v[j][0]  = 0;
            v[j][nx - 1] = 0;
        }

        for (int i = 0; i < nx; i++) {
            u[0][i]  = 0;
            u[ny - 1][i] = 1;
            v[0][i]  = 0;
            v[ny - 1][i] = 0;
        }
        plt::contourf(x, y, p);
        plt::quiver(x, y, u, v);
        plt::pause(.01);
        plt::clf();
    }
    plt::show();
    return 0;
}