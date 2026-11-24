// Exact saved MFEM ND/RT field evaluation. This never solves Maxwell equations.
#include <mfem.hpp>
#include <nlohmann/json.hpp>
#include "utils/constants.hpp"
#include <array>
#include <complex>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>
using json=nlohmann::json;
using namespace mfem;
constexpr double mm_to_m=1e-3;
const double eps0=palace::electromagnetics::epsilon0_;
const double mu0=palace::electromagnetics::mu0_;
struct Fields { std::unique_ptr<GridFunction> er,ei,br,bi; };
Mesh load_saved_mesh(const std::string &fields) {
  const auto path = fields + "/mesh.000000";
  std::ifstream stream(path);
  if (!stream) throw std::runtime_error("Missing saved MFEM mesh: " + path);
  // Refinement preparation or orientation repair can reorder the saved field DOFs.
  return Mesh(stream, /*generate_edges=*/1, /*refine=*/0, /*fix_orientation=*/false);
}
std::unique_ptr<GridFunction> load(Mesh &m,const std::string &p) {
  std::ifstream s(p);if(!s)throw std::runtime_error("Missingfield:"+p);
  return std::make_unique<GridFunction>(&m,s);
}
json integrate(Mesh &mesh,GridFunction &er,GridFunction &ei,GridFunction &br,GridFunction &bi,
               int order,const json &windows,const std::string &cell_path,bool fixture=false) {
  double ee=0,em=0;std::map<int,double> ee_mat,em_mat;std::map<std::string,double> region;
  std::ofstream cells;if(!cell_path.empty()){cells.open(cell_path);cells<<std::setprecision(17)<<"element,attribute,x_mm,y_mm,z_mm,volume_mm3,Ee_solver_J,Em_solver_J\n";}
  Vector vr(3),vi(3),mr(3),mi(3),xyz(3),centre(3);
  for(int e=0;e<mesh.GetNE();e++) {
    auto *T=mesh.GetElementTransformation(e);int attr=mesh.GetAttribute(e);
    if(!fixture&&attr!=501&&attr!=514)throw std::runtime_error("Unregisteredmaterialattribute");
    const double epsr=(!fixture&&attr==514)?11.45:1.;
    const auto &rule=IntRules.Get(mesh.GetElementBaseGeometry(e),order);double ue=0,um=0,volume=0;
    for(int j=0;j<rule.GetNPoints();j++) {
      const auto &ip=rule.IntPoint(j);T->SetIntPoint(&ip);
      er.GetVectorValue(*T,ip,vr);ei.GetVectorValue(*T,ip,vi);br.GetVectorValue(*T,ip,mr);bi.GetVectorValue(*T,ip,mi);
      double w=ip.weight*T->Weight();volume+=w;
      ue+=.5*eps0*epsr*((vr*vr)+(vi*vi))*w*mm_to_m*mm_to_m*mm_to_m;
      um+=.5/mu0*((mr*mr)+(mi*mi))*w*mm_to_m*mm_to_m*mm_to_m;
    }
    ee+=ue;em+=um;ee_mat[attr]+=ue;em_mat[attr]+=um;
    const auto &mid=Geometries.GetCenter(mesh.GetElementBaseGeometry(e));T->Transform(mid,centre);
    for(const auto &region_def:windows) {
      auto bounds=region_def["bounds_mm"].get<std::vector<double>>();
      if(centre[0]>=bounds[0]&&centre[0]<bounds[1]&&centre[1]>=bounds[2]&&centre[1]<bounds[3]&&centre[2]>=bounds[4]&&centre[2]<=bounds[5])region[region_def["label"].get<std::string>()]+=ue;
    }
    if(cells)cells<<e<<','<<attr<<','<<centre[0]<<','<<centre[1]<<','<<centre[2]<<','<<volume<<','<<ue<<','<<um<<'\n';
  }
  struct Port {double area=0,minx=1e90,maxx=-1e90,integral_r=0,integral_i=0,Et2=0,Ex2=0;int faces=0;};
  std::map<int,Port> ports;
  for(int b=0;b<mesh.GetNBE();b++) {
    int attr=mesh.GetBdrAttribute(b);if(attr!=395&&attr!=402)continue;
    auto &p=ports[attr];p.faces++;Array<int> vs;mesh.GetBdrElementVertices(b,vs);
    for(int k=0;k<vs.Size();k++){const double x=mesh.GetVertex(vs[k])[0];p.minx=std::min(p.minx,x);p.maxx=std::max(p.maxx,x);}
    auto *T=mesh.GetBdrElementTransformation(b);int face_index,orientation;mesh.GetBdrElementFace(b,&face_index,&orientation);
    auto *F=mesh.GetFaceElementTransformations(face_index);if(!F||!F->Elem1)throw std::runtime_error("Missinginternalfaceadjacencymap");
    const auto &rule=IntRules.Get(mesh.GetBdrElementBaseGeometry(b),order);
    for(int j=0;j<rule.GetNPoints();j++) {
      const auto &ip=rule.IntPoint(j);T->SetIntPoint(&ip);
      IntegrationPoint face_ip=Mesh::TransformBdrElementToFace(F->GetGeometryType(),orientation,ip);F->SetAllIntPoints(&face_ip);
      const auto &in=F->GetElement1Transformation().GetIntPoint();
      er.GetVectorValue(*F->Elem1,in,vr);ei.GetVectorValue(*F->Elem1,in,vi);
      Vector normal(3);CalcOrtho(T->Jacobian(),normal);normal/=normal.Norml2();
      const double w=ip.weight*T->Weight()*mm_to_m*mm_to_m;p.area+=w;
      p.integral_r+=vr[0]*w;p.integral_i+=vi[0]*w;
      p.Et2+=((vr*vr)+(vi*vi)-(vr*normal)*(vr*normal)-(vi*normal)*(vi*normal))*w;
      p.Ex2+=(vr[0]*vr[0]+vi[0]*vi[0])*w;
    }
  }
  json port_json=json::array();for(const auto &[attr,p]:ports) {
    const double length=(p.maxx-p.minx)*mm_to_m,width=p.area/length,Rs=50.*width/length;
    const std::complex<double> voltage(p.integral_r/width,p.integral_i/width);
    const double surface=.5*p.Et2/Rs,direction=.5*p.Ex2/Rs,average=.5*std::norm(voltage)/50.;
    port_json.push_back({{"attribute",attr},{"faces",p.faces},{"area_m2",p.area},{"length_m",length},{"width_m",width},{"R_ohm",50.},{"Rs_ohm",Rs},
      {"V_average_real_V",voltage.real()},{"V_average_imag_V",voltage.imag()},{"P_surface_tangential_avg_W",surface},{"P_direction_square_avg_W",direction},
      {"P_average_voltage_avg_W",average},{"P_direction_variance_W",direction-average},{"P_transverse_W",surface-direction}});
  }
  json mats=json::array();for(const auto &[attr,value]:ee_mat)mats.push_back({{"attribute",attr},{"Ee_solver_J",value},{"Em_solver_J",em_mat[attr]}});
  return {{"quadrature_order",order},{"elements",mesh.GetNE()},{"vertices",mesh.GetNV()},{"E_dofs",er.Size()},{"B_dofs",br.Size()},
    {"Ee_solver_J",ee},{"Em_solver_J",em},{"U_total_time_average_J",.5*(ee+em)},{"material_energies",mats},{"region_Ee_solver_J",region},{"ports",port_json}};
}
json benchmark() {
  Mesh mesh=Mesh::MakeCartesian3D(1,1,2,Element::TETRAHEDRON,1.,1.,1.);
  for(int b=0;b<mesh.GetNBE();b++)mesh.GetBdrElement(b)->SetAttribute(901);
  std::vector<std::array<int,3>> internal;
  for(int f=0;f<mesh.GetNumFaces();f++){Array<int> v;mesh.GetFaceVertices(f,v);bool middle=v.Size()==3;for(int k=0;k<v.Size();k++)middle=middle&&std::abs(mesh.GetVertex(v[k])[2]-.5)<1e-12;if(middle)internal.push_back({v[0],v[1],v[2]});}
  for(const auto &v:internal)mesh.AddBdrTriangle(v[0],v[1],v[2],395);mesh.FinalizeTopology();mesh.SetAttributes();
  ND_FECollection nd(1,3);RT_FECollection rt(0,3);FiniteElementSpace ne(&mesh,&nd),re(&mesh,&rt);
  GridFunction er(&ne),ei(&ne),br(&re),bi(&re);ei=0.;bi=0.;Vector B(3);B[0]=.1;B[1]=.2;B[2]=.3;VectorConstantCoefficient bc(B);br.ProjectCoefficient(bc);
  Vector E(3);E=0.;E[0]=2.;VectorConstantCoefficient ec(E);er.ProjectCoefficient(ec);
  auto constant=integrate(mesh,er,ei,br,bi,6,json::array(),"",true);
  VectorFunctionCoefficient affine(3,[](const Vector &x,Vector &v){v.SetSize(3);v[0]=1.-x[1];v[1]=x[0];v[2]=0.;});er.ProjectCoefficient(affine);
  auto linear=integrate(mesh,er,ei,br,bi,6,json::array(),"",true);
  return {{"constant_fixture",constant},{"affine_ND_fixture",linear},{"expected_constant_Ee_J",.5*eps0*4.e-9},{"expected_constant_Em_J",.5/mu0*.14e-9},
    {"expected_constant_port_W",4e-8},{"expected_affine_Ee_J",.5*eps0*(2./3.)*1e-9},{"expected_affine_surface_port_W",(2./3.)*1e-8},{"expected_affine_average_port_W",.25e-8},
    {"scope","Exact representable analytic ND/RT fields on an uncracked INTERNAL port sheet; separately tests surface loss and projected voltage, not a Maxwell solve"}};
}

// Retained common-physical-point API for future complex-vector overlaps.
// This does not substitute scalar energy histograms for vector inner products.
int samples(int argc,char **argv) {
  if(argc!=6)throw std::runtime_error("Usage --sample fields_dir points_csv output_csv mode");
  std::string fields=argv[2];int mode=std::stoi(argv[5]);Mesh mesh=load_saved_mesh(fields);
  std::ostringstream index;index<<std::setw(6)<<std::setfill('0')<<mode;
  auto er=load(mesh,fields+"/E_real_"+index.str()+".gf.000000"),ei=load(mesh,fields+"/E_imag_"+index.str()+".gf.000000");
  std::ifstream input(argv[3]);if(!input)throw std::runtime_error("Missingphysicalpointlist");
  std::vector<std::array<double,3>> points;std::string line;
  while(std::getline(input,line)){std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);std::array<double,3> p;if(row>>p[0]>>p[1]>>p[2])points.push_back(p);}
  DenseMatrix coords(3,points.size());for(int j=0;j<int(points.size());j++)for(int k=0;k<3;k++)coords(k,j)=points[j][k];
  Array<int> ids;Array<IntegrationPoint> ips;int found=mesh.FindPoints(coords,ids,ips,false);
  std::ofstream out(argv[4]);out<<std::setprecision(17)<<"point,x_mm,y_mm,z_mm,found,element,attribute,Ex_real,Ey_real,Ez_real,Ex_imag,Ey_imag,Ez_imag\n";
  Vector vr(3),vi(3);for(int j=0;j<int(points.size());j++){
    out<<j<<','<<points[j][0]<<','<<points[j][1]<<','<<points[j][2]<<','<<(ids[j]>=0?1:0)<<','<<ids[j];
    if(ids[j]>=0){er->GetVectorValue(ids[j],ips[j],vr);ei->GetVectorValue(ids[j],ips[j],vi);out<<','<<mesh.GetAttribute(ids[j]);for(int k=0;k<3;k++)out<<','<<vr[k];for(int k=0;k<3;k++)out<<','<<vi[k];}
    else out<<",,,,,,,";out<<'\n';
  }
  std::cout<<"Sampled "<<found<<" of "<<points.size()<<" physical points"<<std::endl;return found==int(points.size())?0:3;
}

int main(int argc,char **argv) {
  try {
    Mpi::Init(argc,argv);if(Mpi::WorldSize()!=1)throw std::runtime_error("ExactlyoneMPIrankrequired");Device device("cpu");
    if(argc>=2&&std::string(argv[1])=="--sample")return samples(argc,argv);
    if(argc==2&&std::string(argv[1])=="--benchmark"){std::cout<<benchmark().dump(2)<<std::endl;return 0;}
    if(argc!=6)throw std::runtime_error("Usage fields_dir output_dir windows_json order mode");
    std::string fields=argv[1],out=argv[2];int order=std::stoi(argv[4]),mode=std::stoi(argv[5]);std::ifstream def(argv[3]);json windows;def>>windows;
    Mesh mesh=load_saved_mesh(fields);std::ostringstream index;index<<std::setw(6)<<std::setfill('0')<<mode;
    auto er=load(mesh,fields+"/E_real_"+index.str()+".gf.000000"),ei=load(mesh,fields+"/E_imag_"+index.str()+".gf.000000"),
         br=load(mesh,fields+"/B_real_"+index.str()+".gf.000000"),bi=load(mesh,fields+"/B_imag_"+index.str()+".gf.000000");
    auto result=integrate(mesh,*er,*ei,*br,*bi,order,windows,order==6?out+"/cells_mode"+std::to_string(mode)+".csv":"");
    result["mode"]=mode;result["epsilon0_F_per_m"]=eps0;result["mu0_H_per_m"]=mu0;
    std::ofstream output(out+"/mode"+std::to_string(mode)+"_q"+std::to_string(order)+".json");output<<result.dump(2)<<'\n';
    std::cout<<"Completed mode "<<mode<<" quadrature "<<order<<std::endl;return 0;
  }catch(const std::exception &e){std::cerr<<e.what()<<std::endl;return 2;}
}
