# Open Source Face Image Quality (OFIQ)

The __OFIQ__ (Open Source Face Image Quality) is a software library for computing quality 
aspects of a facial image. OFIQ is written in the C/C++ programming language.
OFIQ is the reference implementation for the ISO/IEC 29794-5 international
standard; see [https://bsi.bund.de/dok/OFIQ-e](https://bsi.bund.de/dok/OFIQ-e).

## License
Before using __OFIQ__ or distributing parts of __OFIQ__ one should have a look
on OFIQ's license and the license of its dependencies: [LICENSE.md](LICENSE.md)
  
## Getting started
For a tutorial on how to compile and operate OFIQ, see [here](BUILD.md).
For a tutorial on how to compile and operatate OFIQ on mobile platforms, 
see [here](mobile/BUILD.md).

## Reference manual
A full documentation of __OFIQ__ including compilation, configuration and a comprehensive doxygen 
documentation of the C/C++ API is contained in the reference manual: 
see [doc/refman.pdf](doc/refman.pdf).

## Known issues
For a list of known issues, see [here](ISSUES.md)

## Acknowledgements

OFIQ features some quality measures that perform among the best as per NIST 
evaluation (at the time of submission); see 
[evaluation report](https://pages.nist.gov/frvt/reports/quality_sidd/frvt_quality_sidd_report.pdf).

Especially, we would like acknowledge of the following contributors:
* Qiang Meng, Shichao Zhao, Zhida Huang, Feng Zhou who trained 
[MagFace](https://github.com/IrvingMeng/MagFace) which is the basis of OFIQ's unified quality 
assessment. MagFace enables one of the best unified quality assessments. It is open source
and can well compete with commercial solutions.
* Xiangnan Yin and Liming Chen who 
provided [FaceExtraction](https://github.com/face3d0725/FaceExtraction) which is the basis of 
OFIQ's occlusion-aware segmentation.

