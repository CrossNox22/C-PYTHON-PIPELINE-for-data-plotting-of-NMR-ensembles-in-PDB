#define EIGEN_DONT_ALIGN_STATICALLY 1
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <Eigen/Dense>
#include <Eigen/SVD>

using namespace Eigen;
using namespace std;
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --- PHYSICAL CONSTANTS ---
const double PROBE_RADIUS = 1.4; // Water probe radius (Angstrom)
const double ATOM_RADIUS = 1.8;  // Approximate Alpha Carbon (CA) radius
const int NUM_POINTS = 96;       // Points on sphere (Balance between precision and speed)

struct Frame {
    int n_atoms;
    MatrixXd coords;
};

// --- UTILITY FUNCTIONS ---

void centerMolecule(MatrixXd& coords) {
    coords.rowwise() -= coords.colwise().mean();
}

// Computes optimal rotation matrix using Kabsch algorithm
Matrix3d computeKabschRotation(const MatrixXd& P, const MatrixXd& Q) {
    Matrix3d H = P.transpose() * Q;
    JacobiSVD<Matrix3d> svd(H, ComputeFullU | ComputeFullV);
    Matrix3d V = svd.matrixV();
    Matrix3d U = svd.matrixU();
    Matrix3d R = V * U.transpose();
    
    // Correction for reflection
    if (R.determinant() < 0) {
        V.col(2) *= -1;
        R = V * U.transpose();
    }
    return R;
}

vector<Frame> readTrajectory(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) throw runtime_error("File not found: " + filename);
    
    vector<Frame> frames;
    while (true) {
        int n;
        if (!(file >> n)) break;
        
        string dummy, comment;
        getline(file, dummy); getline(file, comment);
        
        Frame f; 
        f.n_atoms = n; 
        f.coords.resize(n, 3);
        
        for (int i = 0; i < n; ++i) {
            string s; double x, y, z;
            file >> s >> x >> y >> z;
            f.coords(i, 0) = x; f.coords(i, 1) = y; f.coords(i, 2) = z;
        }
        frames.push_back(f);
    }
    return frames;
}

// --- SHRAKE-RUPLEY ALGORITHM (SASA) ---

// Generates equidistant points on a unit sphere using the Golden Spiral method
vector<Vector3d> generateSpherePoints(int n) {
    vector<Vector3d> points;
    double inc = M_PI * (3.0 - sqrt(5.0));
    double off = 2.0 / n;
    
    for (int k = 0; k < n; ++k) {
        double y = k * off - 1.0 + (off / 2.0);
        double r = sqrt(1.0 - y * y);
        double phi = k * inc;
        points.push_back(Vector3d(cos(phi) * r, y, sin(phi) * r));
    }
    return points;
}

vector<double> computeAverageSASA(const vector<Frame>& frames) {
    int n_atoms = frames[0].n_atoms;
    int n_frames = frames.size();
    vector<double> avg_sasa(n_atoms, 0.0);
    
    // Generate reference sphere once
    vector<Vector3d> sphere_points = generateSpherePoints(NUM_POINTS);
    double effective_radius = ATOM_RADIUS + PROBE_RADIUS;
    double area_per_point = (4.0 * M_PI * pow(effective_radius, 2)) / NUM_POINTS;

    for (const auto& f : frames) {
        for (int i = 0; i < n_atoms; ++i) {
            int accessible_points = 0;
            Vector3d center = f.coords.row(i);

            // Check surface points against neighbors
            for (const auto& pt : sphere_points) {
                Vector3d surface_point = center + (pt * effective_radius);
                bool buried = false;

                for (int j = 0; j < n_atoms; ++j) {
                    if (i == j) continue;
                    
                    double dist_sq = (surface_point - f.coords.row(j).transpose()).squaredNorm();
                    
                    if (dist_sq < pow(effective_radius, 2)) {
                        buried = true;
                        break;
                    }
                }
                if (!buried) accessible_points++;
            }
            avg_sasa[i] += accessible_points * area_per_point;
        }
    }

    // Average over frames
    for (double& val : avg_sasa) val /= n_frames;
    return avg_sasa;
}

// --- MAIN EXECUTION ---

int main(int argc, char* argv[]) {
    try {
        if (argc != 4) {
            cerr << "Usage: analyzer <input.xyz> <out_rmsf.txt> <out_sasa.txt>" << endl;
            return 1;
        }
        string inputFile = argv[1];
        string outRmsf = argv[2];
        string outSasa = argv[3];

        vector<Frame> frames = readTrajectory(inputFile);
        if (frames.empty()) return 1;

        // 1. Structural Alignment and RMSF Calculation
        Frame& ref = frames[0];
        centerMolecule(ref.coords);
        MatrixXd meanCoords = MatrixXd::Zero(ref.n_atoms, 3);
        
        // Superimpose all frames to the first frame
        for (size_t i = 1; i < frames.size(); ++i) {
            centerMolecule(frames[i].coords);
            Matrix3d R = computeKabschRotation(ref.coords, frames[i].coords);
            frames[i].coords = frames[i].coords * R;
        }
        
        // Compute Mean Structure
        for (const auto& f : frames) meanCoords += f.coords;
        meanCoords /= frames.size();

        // Compute RMSF
        ofstream f_rmsf(outRmsf);
        for (int i = 0; i < ref.n_atoms; ++i) {
            double sum_sq = 0.0;
            for (const auto& f : frames) sum_sq += (f.coords.row(i) - meanCoords.row(i)).squaredNorm();
            f_rmsf << fixed << setprecision(4) << sqrt(sum_sq / frames.size()) << "\n";
        }

        // 2. SASA Calculation
        vector<double> sasa = computeAverageSASA(frames);
        ofstream f_sasa(outSasa);
        for (double val : sasa) f_sasa << fixed << setprecision(4) << val << "\n";

    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    return 0;
}