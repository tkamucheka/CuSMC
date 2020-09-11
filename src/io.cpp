#ifndef __IO_CPP
#define __IO_CPP

#include <string>
#include "../inst/include/io.hpp"

void writeOutput(Eigen::VectorXd(*y_t), Eigen::VectorXd(*w_t), 
                 Eigen::VectorXd(**post_x_t), const dim_t N, 
                 const dim_t d, const dim_t timeSteps, dim_t p)
{
  std::ofstream file_y_time1_csv("y_t.csv");
  std::ofstream file_x_time1_csv("x_t_N" + std::to_string(p) + ".csv");
  file_y_time1_csv << "y0,y1,w" << std::endl;
  file_x_time1_csv << "x0,x1,w" << std::endl;


  for (unsigned i = 0; i < timeSteps; ++i)
  {
    file_y_time1_csv << y_t[i][0] << "," << y_t[i][1] 
                     << std::endl;
    file_x_time1_csv << post_x_t[i][p][0] << "," << post_x_t[i][p][1]
                     << "," << w_t[i][0] << std::endl;
  }

  std::ofstream file_x_theta1_csv("x_t1.csv");
  file_x_theta1_csv << "x_t" << std::endl;

  int time = 1;
  for (unsigned i = 0; i < N; ++i)
  {
    file_x_theta1_csv << post_x_t[time][i][0] << "," << post_x_t[time][i][1]
                      << std::endl;
  }
}

void writeOutput_ysim(Eigen::VectorXd(*prior_x_t), Eigen::VectorXd(*y_t),
                 Eigen::VectorXd(*w_t), Eigen::VectorXd(**post_x_t),
                 const dim_t N, const dim_t d, const dim_t timeSteps)
{
  std::ofstream file_prior_x_txt("prior_x_t.txt");
  std::ofstream file_post_x_txt("post_x_t.txt");
  for (unsigned t = 0; t < timeSteps; ++t)
  {
    file_prior_x_txt << prior_x_t[t].transpose() << std::endl;
    file_post_x_txt << post_x_t[t][0].transpose() << std::endl;
  }

  std::ofstream file_x_in_time1_csv("x_in_time1.csv");
  std::ofstream file_y_time1_csv("y_time1.csv");
  //std::ofstream file_y_only_csv("y_t.csv");
  std::ofstream file_x_time1_csv("x_time1.csv");
  file_x_in_time1_csv << "y0,y1,w" << std::endl;
  file_y_time1_csv << "y0,y1,w" << std::endl;
  //file_y_only_csv << "y0,y1" << std::endl;
  file_x_time1_csv << "x0,x1,w" << std::endl;

  int theta = 0;
  for (unsigned i = 0; i < timeSteps; ++i)
  {
    file_x_in_time1_csv << prior_x_t[i][0] << "," << prior_x_t[i][1]
                        << "," << w_t[i][0] << std::endl;
    file_y_time1_csv << y_t[i][0] << "," << y_t[i][1] << "," << w_t[i][0]
                     << std::endl;
    //file_y_only_csv << y_t[i][0] << "," << y_t[i][1] << std::endl;
    file_x_time1_csv << post_x_t[i][theta][0] << "," << post_x_t[i][theta][1]
                     << "," << w_t[i][0] << std::endl;
  }

  std::ofstream file_x_theta1_csv("x_theta1.csv");
  file_x_theta1_csv << "x_t" << std::endl;

  int time = 1;
  for (unsigned i = 0; i < N; ++i)
  {
    file_x_theta1_csv << post_x_t[time][i][0] << "," << post_x_t[time][i][1]
                      << std::endl;
  }
}
#endif
