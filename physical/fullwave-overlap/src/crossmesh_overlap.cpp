// Read-only common-physical-point complex ND Gram. No Maxwell solve.
#define main retained_exact_fields_main
#include "exact_fields.cpp"
#undef main
#include <algorithm>
#include <numeric>
struct Tet {double lo[3],hi[3],o[3],inv[3][3];int attr;};
struct Node {double lo[3],hi[3];int a,b,left=-1,right=-1;};
struct Locator {
 Mesh &mesh;std::vector<Tet> t;std::vector<int> ids;std::vector<Node> nodes;
 Locator(Mesh &m):mesh(m){
  t.resize(m.GetNE());ids.resize(m.GetNE());std::iota(ids.begin(),ids.end(),0);
  for(int e=0;e<m.GetNE();e++){
   Array<int> vs;m.GetElementVertices(e,vs);if(vs.Size()!=4)throw std::runtime_error("Only affine tetrahedra supported");
   auto &q=t[e];q.attr=m.GetAttribute(e);for(int k=0;k<3;k++){q.o[k]=m.GetVertex(vs[0])[k];q.lo[k]=1e99;q.hi[k]=-1e99;for(int v:vs){q.lo[k]=std::min(q.lo[k],m.GetVertex(v)[k]);q.hi[k]=std::max(q.hi[k],m.GetVertex(v)[k]);}}
   DenseMatrix M(3);for(int j=0;j<3;j++)for(int k=0;k<3;k++)M(k,j)=m.GetVertex(vs[j+1])[k]-q.o[k];M.Invert();for(int k=0;k<3;k++)for(int j=0;j<3;j++)q.inv[k][j]=M(k,j);
   auto *T=m.GetElementTransformation(e);for(int v=0;v<4;v++){IntegrationPoint p;p.Set3(v==1,v==2,v==3);Vector x(3);T->Transform(p,x);for(int k=0;k<3;k++)if(std::abs(x[k]-m.GetVertex(vs[v])[k])>1e-10)throw std::runtime_error("Non-affine saved geometry");}
  }build(0,ids.size());
 }
 int build(int a,int b){int n=nodes.size();nodes.emplace_back();Node q;q.a=a;q.b=b;for(int k=0;k<3;k++){q.lo[k]=1e99;q.hi[k]=-1e99;for(int j=a;j<b;j++){q.lo[k]=std::min(q.lo[k],t[ids[j]].lo[k]);q.hi[k]=std::max(q.hi[k],t[ids[j]].hi[k]);}}nodes[n]=q;
  if(b-a>12){int k=0;for(int d=1;d<3;d++)if(q.hi[d]-q.lo[d]>q.hi[k]-q.lo[k])k=d;int mid=(a+b)/2;std::nth_element(ids.begin()+a,ids.begin()+mid,ids.begin()+b,[&](int i,int j){return t[i].lo[k]+t[i].hi[k]<t[j].lo[k]+t[j].hi[k];});int l=build(a,mid),r=build(mid,b);nodes[n].left=l;nodes[n].right=r;}return n;
 }
 int find(const Vector &x,IntegrationPoint &ip,int n=0)const {const auto &q=nodes[n];for(int k=0;k<3;k++)if(x[k]<q.lo[k]-1e-12||x[k]>q.hi[k]+1e-12)return -1;
  if(q.left>=0){int a=find(x,ip,q.left);return a>=0?a:find(x,ip,q.right);}for(int j=q.a;j<q.b;j++){int e=ids[j];const auto &v=t[e];double p[3]={};for(int k=0;k<3;k++)for(int d=0;d<3;d++)p[k]+=v.inv[k][d]*(x[d]-v.o[d]);if(p[0]>=-1e-10&&p[1]>=-1e-10&&p[2]>=-1e-10&&p[0]+p[1]+p[2]<=1+1e-10){ip.Set3(p[0],p[1],p[2]);return e;}}return -1;
 }
};
json cross(Mesh &a,Mesh &b,std::vector<std::unique_ptr<GridFunction>> &r,std::vector<std::unique_ptr<GridFunction>> &im,int na,int ng){
 Locator locator(b);int n=r.size();std::vector<double> gr(n*n),gi(n*n);std::vector<std::array<double,3>> vr(n),vi(n);Vector x(3),v(3),w(3);double missed=0,matbad=0,total=0;long points=0;
 const auto &line=IntRules.Get(Geometry::SEGMENT,2*ng-1);if(line.GetNPoints()!=ng)throw std::runtime_error("Unexpected Gauss rule");
 for(int e=0;e<a.GetNE();e++){auto *T=a.GetElementTransformation(e);int attr=a.GetAttribute(e);if(attr!=501&&attr!=514)throw std::runtime_error("Unregistered material");
  for(int i=0;i<ng;i++)for(int j=0;j<ng;j++)for(int k=0;k<ng;k++){
   auto u=line.IntPoint(i),v0=line.IntPoint(j),w0=line.IntPoint(k);IntegrationPoint ip;ip.Set3(u.x,(1-u.x)*v0.x,(1-u.x)*(1-v0.x)*w0.x);T->SetIntPoint(&ip);T->Transform(ip,x);
   double weight=u.weight*v0.weight*w0.weight*(1-u.x)*(1-u.x)*(1-v0.x)*T->Weight()*1e-9;total+=weight;points++;IntegrationPoint bp;int be=locator.find(x,bp);
   if(be<0){missed+=weight;continue;}if(b.GetAttribute(be)!=attr){matbad+=weight;continue;}
   for(int f=0;f<n;f++){if(f<na){r[f]->GetVectorValue(*T,ip,v);im[f]->GetVectorValue(*T,ip,w);}else{r[f]->GetVectorValue(be,bp,v);im[f]->GetVectorValue(be,bp,w);}for(int d=0;d<3;d++){vr[f][d]=v[d];vi[f][d]=w[d];}}
   weight*=eps0*(attr==514?11.45:1.);for(int f=0;f<n;f++)for(int g=f;g<n;g++)for(int d=0;d<3;d++){gr[f*n+g]+=weight*(vr[f][d]*vr[g][d]+vi[f][d]*vi[g][d]);gi[f*n+g]+=weight*(vr[f][d]*vi[g][d]-vi[f][d]*vr[g][d]);}
  }
 }
 for(int i=0;i<n;i++)for(int j=0;j<i;j++){gr[i*n+j]=gr[j*n+i];gi[i*n+j]=-gi[j*n+i];}json matrix=json::array();for(int i=0;i<n;i++){json row=json::array();for(int j=0;j<n;j++)row.push_back({gr[i*n+j],gi[i*n+j]});matrix.push_back(row);}
 return {{"Gram_complex_re_im",matrix},{"count_A",na},{"count_B",n-na},{"Duffy_Gauss_points_per_axis",ng},{"positive_weight_points",points},{"domain_volume_m3",total},{"unlocated_volume_fraction",missed/total},{"material_mismatch_volume_fraction",matbad/total},{"scope","Actual complex ND fields evaluated at common physical interior points; positive Duffy weights, affine saved-tetra BVH; no H1 projection, no frequency sorting"}};
}
int main(int argc,char **argv){try{Mpi::Init(argc,argv);Device device("cpu");if(argc==2&&std::string(argv[1])=="--benchmark"){
 Mesh a=Mesh::MakeCartesian3D(1,1,1,Element::TETRAHEDRON,1.,1.,1.),b=Mesh::MakeCartesian3D(2,1,1,Element::TETRAHEDRON,1.,1.,1.);for(int e=0;e<a.GetNE();e++)a.GetElement(e)->SetAttribute(501);for(int e=0;e<b.GetNE();e++)b.GetElement(e)->SetAttribute(501);ND_FECollection nd(1,3);FiniteElementSpace sa(&a,&nd),sb(&b,&nd);std::vector<std::unique_ptr<GridFunction>> r,im;
 for(int f=0;f<4;f++){auto *sp=f<2?&sa:&sb;r.push_back(std::make_unique<GridFunction>(sp));im.push_back(std::make_unique<GridFunction>(sp));int mode=f%2;VectorFunctionCoefficient cr(3,[mode](const Vector &x,Vector &v){v.SetSize(3);v=0.;v[0]=mode?1.-x[1]:2.;v[1]=mode?x[0]:0.;});VectorFunctionCoefficient ci(3,[mode](const Vector &x,Vector &v){v.SetSize(3);v=0.;v[2]=mode?0.:1.;});r.back()->ProjectCoefficient(cr);im.back()->ProjectCoefficient(ci);}
 json out={{"q3",cross(a,b,r,im,2,3)},{"q4",cross(a,b,r,im,2,4)},{"expected_norms",{5.*eps0*1e-9,(2./3.)*eps0*1e-9}},{"expected_cross",eps0*1e-9}};std::cout<<out.dump(2)<<"\n";return 0;
 }if(Mpi::WorldSize()!=1||argc!=7)throw std::runtime_error("Usage fieldsA countA fieldsB countB ng output");std::ifstream fa(std::string(argv[1])+"/mesh.000000"),fb(std::string(argv[3])+"/mesh.000000");Mesh a(fa,1,1,false),b(fb,1,1,false);int na=std::stoi(argv[2]),nb=std::stoi(argv[4]);std::vector<std::unique_ptr<GridFunction>> r,im;for(int f=0;f<na+nb;f++){std::string dir=f<na?argv[1]:argv[3];int idx=f<na?f+1:f-na+1;std::ostringstream s;s<<std::setw(6)<<std::setfill('0')<<idx;Mesh &m=f<na?a:b;r.push_back(load(m,dir+"/E_real_"+s.str()+".gf.000000"));im.push_back(load(m,dir+"/E_imag_"+s.str()+".gf.000000"));}auto out=cross(a,b,r,im,na,std::stoi(argv[5]));std::ofstream f(argv[6]);f<<out.dump(2)<<'\n';std::cout<<"Cross-mesh Gram integrated\n";return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}
