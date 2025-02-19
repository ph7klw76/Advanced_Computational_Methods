# Reorganization between singlet and triplet

Calculate the  opt singlet
```python
! DEF2-SVP OPT CPCM(toluene)  # Opt Singlet excited Geo
%TDDFT  NROOTS  2
        IROOT  2
        IROOTMULT SINGLET
        LRCPCM True
END
%method
        method dft
        functional HYB_GGA_XC_LRC_WPBEH
	ExtParamXC "_omega" 0.0645
end           
%maxcore 6000
%pal nprocs 16 end
* XYZFILE 0 1 235-tPBisICz.xyz
```


claculate singlet energy based on opt triplet geometry
```python
! DEF2-SVP OPT CPCM(toluene)  # Opt 2nd triplet excited Geo
%TDDFT  NROOTS  2
        IROOT  2
        IROOTMULT TRIPLET
        LRCPCM True
END
%scf
  MaxIter 300
end
%method
        method dft
        functional HYB_GGA_XC_LRC_WPBEH
	ExtParamXC "_omega" 0.0505
end           
%maxcore 5000
%pal nprocs 16 end
* XYZFILE 0 1 EHBIPOAc0.050460446612374865.xyz
$new_job
!DEF2-SVP OPT CPCM(toluene)    # Cal Singlet excited Energy based on 2nd Opt triplet Geo
%TDDFT  NROOTS  1
        IROOTMULT SINGLET
END
%method
        method dft
        functional HYB_GGA_XC_LRC_WPBEH
	ExtParamXC "_omega" 0.0505
end           
%maxcore 5000
%pal nprocs 16 end
* XYZFILE 0 1 T.xyz
```
