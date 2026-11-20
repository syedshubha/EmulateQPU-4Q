"""Physical flux/current bath, declared coherent partial-secular Markov model.

Not a unique/global Davies construction. Grouping is by changed local level
pairs and photon-number change, never by exact conditional transition energy.
Thus unresolved spectator-conditioned cavity lines remain one coherent jump.
"""
from __future__ import annotations
import numpy as np
from scipy.sparse import coo_matrix,diags

def series_conductance(omega_per_ns,Cc_fF,R_ohm=25.):
    omega=np.asarray(omega_per_ns,float);tau=R_ohm*1e-6*Cc_fF
    return R_ohm*1e-6*omega**2*Cc_fF**2/(1+(omega*tau)**2)

def physical_flux_bath(E_GHz,labels,photons,Phi_dimensionless,*,fr_GHz,
                       Cmodal_fF,Cc_fF,R_ohm=25.,frame_GHz=None,
                       external_E_GHz=None,external_Phi_dimensionless=None,
                       conductance=None,coefficient_cutoff=1e-14,
                       positive_gap_GHz=1e-10):
    """Return physical-port collapse groups and a trace-preserving sink.

    Phi_dimensionless = Phi_physical/Phi_zpf, Hermitian; Phi_zpf² =
    hbar/(2*Cmodal*omega_r). Rates are angular-frequency FDR rates/ns:
    (omega_ij/omega_r)*G(omega_ij)/Cmodal*|Phi/Phi_zpf|².
    `conductance`, when supplied by a controlled fixture, returns fF/ns.
    Actual production uses the declared seriesCc–R conductance above.
    """
    E=np.asarray(E_GHz,float);labels=np.asarray(labels,int);n=np.asarray(photons,int);P=np.asarray(Phi_dimensionless,complex)
    if labels.ndim==1:labels=labels[:,None]
    d=len(E)
    if P.shape!=(d,d) or len(labels)!=d or n.shape!=(d,):raise ValueError('Bathshapemismatch')
    if fr_GHz<=0 or Cmodal_fF<=0 or Cc_fF<0 or R_ohm<=0:raise ValueError('Positivepassiveparametersrequired')
    if np.max(abs(P-P.conj().T))>1e-10:raise ValueError('PhysicalfluxmustbeHermitian')
    G=conductance if conductance is not None else lambda omega:series_conductance(omega,Cc_fF,R_ohm)
    gap=E[None,:]-E[:,None];mask=(gap>positive_gap_GHz)&(abs(P)>=coefficient_cutoff);ii,jj=np.nonzero(mask)
    weights=(gap[ii,jj]/fr_GHz)*np.asarray(G(2*np.pi*gap[ii,jj]),float)/Cmodal_fF
    if np.any(weights<0) or not np.all(np.isfinite(weights)):raise ValueError('PassivePSDmustbenonnegativefinite')
    amps=np.sqrt(weights)*P[ii,jj];groups={};rates=np.zeros(d)
    for i,j,amplitude in zip(ii,jj,amps):
        changes=tuple((k,int(labels[j,k]),int(labels[i,k])) for k in range(labels.shape[1]) if labels[j,k]!=labels[i,k]);key=(changes,int(n[i]-n[j]))
        groups.setdefault(key,[]).append((int(i),int(j),complex(amplitude)));rates[j]+=abs(amplitude)**2
    channels=[]
    for key,entries in groups.items():
        dst=np.array([x[0] for x in entries],int);src=np.array([x[1] for x in entries],int);amplitude=np.array([x[2] for x in entries],complex)
        if len(set(dst))!=len(dst) or len(set(src))!=len(src):raise ValueError('Groupnotapartialpermutation')
        spread=0.
        if frame_GHz is not None:
            F=np.asarray(frame_GHz,float);spread=float(np.ptp(F[dst]-F[src]))
            if spread>1e-9:raise ValueError('Groupdoesnotshareoneframephase')
        channels.append(dict(key=key,dst=dst,src=src,amplitude=amplitude,frame_spread_GHz=spread))
    external=np.zeros(d)
    if external_E_GHz is not None:
        Ex=np.asarray(external_E_GHz,float);Px=np.asarray(external_Phi_dimensionless,complex)
        if Px.shape!=(len(Ex),d):raise ValueError('Externalbathshapemismatch')
        gx=E[None,:]-Ex[:,None];valid=gx>positive_gap_GHz;safe=np.maximum(gx,0.);weight=(safe/fr_GHz)*np.asarray(G(2*np.pi*safe),float)/Cmodal_fF
        if np.any(weight<0):raise ValueError('NegativeexternalPSD')
        external=np.sum(np.where(valid,weight*abs(Px)**2,0.),axis=0)
    return dict(groups=channels,selected_loss_rates=rates,external_rates=external,
                metadata=dict(group_count=len(channels),positive_transitions=len(ii),
                    temperature_K=0.,PSD='SII+(omega)=2*hbar*omega*ReYseries(omega)',
                    operator='PhysicalPhi/Phi_zpf;NOTbareX',
                    grouping='Changedlocallevelpairs+photonchange,coherentsmoothfrequencyweights;NOTexactDavies',
                    coefficient_cutoff=coefficient_cutoff,positive_gap_GHz=positive_gap_GHz))

def bath_liouvillian(bath,d,include_sink=True):
    """Row-major density vectorization; collective groups retain coherence."""
    D=d+int(include_sink);rows=[];cols=[];data=[];loss=np.r_[bath['selected_loss_rates']+bath['external_rates'],np.zeros(int(include_sink))]
    for c in bath['groups']:
        i,j,a=c['dst'],c['src'],c['amplitude'];rows.extend((i[:,None]*D+i[None,:]).ravel());cols.extend((j[:,None]*D+j[None,:]).ravel());data.extend((a[:,None]*a[None,:].conj()).ravel())
    if include_sink:
        for j,g in enumerate(bath['external_rates']):
            if g:rows.append(D*D-1);cols.append(j*D+j);data.append(g)
    elif np.any(bath['external_rates']):raise ValueError('Cannotdropnonzeroexternallosswithoutasink')
    L=coo_matrix((np.array(data,complex),(rows,cols)),shape=(D*D,D*D)).tocsr();L+=diags((-.5*(loss[:,None]+loss[None,:])).ravel(),format='csr')
    return L
