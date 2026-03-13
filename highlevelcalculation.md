
Ground State
! r2SCAN-3c VeryTightOpt CPCM(Toluene) TIGHTSCF
%maxcore 4000
%pal nprocs 32 end
* XYZFILE 0 1 174-2PTZBN.xyz

! SOS-WB2GP-PLYP DEF2-TZVP(-f) def2-TZVP/C tightSCF Opt NumGrad
%tddft
   nroots 2
   iroot 1
   TRIPLETS False
   tda True
end
%maxcore 4000
%pal nprocs 32 end
* XYZFILE 0 1 174-2PTZBN.xyz

