/* The Image Registration Toolkit (IRTK)
 *
 * Copyright 2008-2015 Imperial College London
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

#ifndef _IRTKIMAGEREGISTRATIONFILTER_H
#define _IRTKIMAGEREGISTRATIONFILTER_H

#include <irtkRegistrationFilter.h>

#include <irtkImage.h>
#include <irtkImageFunction.h>
#include <irtkRegisteredImage.h>
#include <irtkRegistrationEnergy.h>
#include <irtkLocalOptimizer.h>
#include <irtkEventDelegate.h>

#ifdef HAS_VTK
#  include <vtkSmartPointer.h>
#  include <vtkPointSet.h>
#  include <irtkRegisteredPointSet.h>
#endif


/**
 * Generic registration filter
 */
class irtkGenericRegistrationFilter : public irtkRegistrationFilter
{
  irtkObjectMacro(irtkGenericRegistrationFilter);

  friend class irtkRegistrationEnergyParser;
  friend class irtkGenericRegistrationLogger;
  friend class irtkGenericRegistrationDebugger;

  // ---------------------------------------------------------------------------
  // Types
public:

  /// Type of resolution pyramid images
  typedef irtkRegisteredImage::InputImageType         ResampledImageType;

  /// List type storing images for one resolution pyramid level
  typedef vector<ResampledImageType>                  ResampledImageList;

  /// Scalar type of resolution pyramid images
  typedef ResampledImageType::VoxelType               VoxelType;

  /// Type of cached displacement field
  typedef irtkGenericImage<double>                    DisplacementImageType;

  /// Structure storing information about transformation instance
  struct TransformationInfo
  {
    double _Exponent;

    TransformationInfo(double e = .0) : _Exponent(e) {}

    bool operator ==(const TransformationInfo &other) const
    {
      return fequal(_Exponent, other._Exponent, 1e-3);
    }

    bool operator !=(const TransformationInfo &other) const
    {
      return !(*this == other);
    }

    static TransformationInfo Identity()
    {
      return TransformationInfo(.0);
    }

    static TransformationInfo Full()
    {
      return TransformationInfo(1.0);
    }

    static TransformationInfo Inverse()
    {
      return TransformationInfo(-1.0);
    }

    bool IsForwardTransformation() const
    {
      return _Exponent > .0;
    }

    bool IsBackwardTransformation() const
    {
      return _Exponent < .0;
    }

    bool IsIdentity() const
    {
      return fequal(_Exponent, .0, 1e-3);
    }

    operator bool() const
    {
      return !IsIdentity();
    }
  };

  /// Structure storing information of a used image similarity term parsed
  /// from the registration energy function string
  struct ImageSimilarityInfo
  {
    irtkSimilarityMeasure _Measure;              ///< Type of similarity measure
    bool                  _DefaultSign;          ///< Whether to use default sign of similarity
    string                _Name;                 ///< Name of similarity term
    double                _Weight;               ///< Weight of similarity term
    int                   _TargetIndex;          ///< Index of target image
    TransformationInfo    _TargetTransformation; ///< Target transformation identifier
    int                   _SourceIndex;          ///< Index of source image
    TransformationInfo    _SourceTransformation; ///< Source transformation identifier

    bool IsSymmetric() const
    {
      return _TargetTransformation && _SourceTransformation;
    }
  };

  /// Structure storing information of a used point set distance term parsed
  /// from the registration energy function string
  struct PointSetDistanceInfo
  {
    irtkPointSetDistanceMeasure _Measure;     ///< Measure of polydata distance
    bool               _DefaultSign;          ///< Whether to use default sign of distance measure
    string             _Name;                 ///< Name of polydata distance term
    double             _Weight;               ///< Weight of polydata distance term
    int                _TargetIndex;          ///< Index of target data set
    TransformationInfo _TargetTransformation; ///< Target transformation identifier
    int                _SourceIndex;          ///< Index of source data set
    TransformationInfo _SourceTransformation; ///< Source transformation identifier

    bool IsSymmetric() const
    {
      return _TargetTransformation && _SourceTransformation;
    }
  };

  /// Structure storing information of a used point set constraint term parsed
  /// from the registration energy function string
  struct PointSetConstraintInfo
  {
    irtkPointSetConstraintMeasure _Measure; ///< Type of constraint/internal forces
    string             _Name;           ///< Name of constraint
    double             _Weight;         ///< Weight of constraint
    int                _PointSetIndex;  ///< Index of input point set object
    int                _RefImageIndex;
    int                _RefPointSetIndex;
    TransformationInfo _Transformation; ///< Point set transformation identifier
  };

  /// Structure storing information of a used constraint term parsed from the
  /// registration energy function string
  struct ConstraintInfo
  {
    irtkConstraintMeasure _Measure; ///< Type of constraint
    string                _Name;    ///< Name of constraint
    double                _Weight;  ///< Weight of constraint
  };

  /// Structure storing information about cached displacements
  struct DisplacementInfo
  {
    int                       _DispIndex;      ///< Index of cached displacement field
    double                    _InputTime;      ///< Time of untransformed data
    irtkImageAttributes       _Domain;         ///< Domain on which it is defined
    const irtkTransformation *_Transformation; ///< Corresponding transformation instance
  };

  /// Structure storing information about transformed output point set
  struct PointSetOutputInfo
  {
    int                _InputIndex;
    bool               _InitialUpdate;
    TransformationInfo _Transformation;
  };

  // ---------------------------------------------------------------------------
  // Attributes

  /// Number of resolution levels
  irtkPublicAttributeMacro(int, NumberOfLevels);

  /// Multi-level transformation mode
  irtkPublicAttributeMacro(irtkMFFDMode, MultiLevelMode);

  /// Transformation model
  irtkPublicAttributeMacro(vector<irtkTransformationModel>, TransformationModel);

  /// Interpolation mode
  irtkPublicAttributeMacro(irtkInterpolationMode, InterpolationMode);

  /// Extrapolation mode
  irtkPublicAttributeMacro(irtkExtrapolationMode, ExtrapolationMode);

  /// Whether to precompute image derivatives or compute them on the fly
  irtkPublicAttributeMacro(bool, PrecomputeDerivatives);

  /// Default similarity measure
  irtkPublicAttributeMacro(irtkSimilarityMeasure, SimilarityMeasure);

  /// Default polydata distance measure
  irtkPublicAttributeMacro(irtkPointSetDistanceMeasure, PointSetDistanceMeasure);

  /// Optimization method
  irtkPublicAttributeMacro(irtkOptimizationMethod, OptimizationMethod);

  /// Normalize weights of energy function terms
  irtkPublicAttributeMacro(bool, NormalizeWeights);

  /// Initial guess of optimal transformation
  irtkPublicAggregateMacro(const irtkTransformation, InitialGuess);

  /// Whether to merge initial global transformation into (local) transformation
  irtkPublicAttributeMacro(bool, MergeGlobalAndLocalTransformation);

  /// Whether to allow x coordinate transformation
  irtkPublicAttributeMacro(bool, RegisterX);

  /// Whether to allow y coordinate transformation
  irtkPublicAttributeMacro(bool, RegisterY);

  /// Whether to allow z coordinate transformation
  irtkPublicAttributeMacro(bool, RegisterZ);

  /// Mask which defines where to evaluate the energy function
  irtkPublicAggregateMacro(irtkBinaryImage, Domain);

  /// Whether to adaptively remesh surfaces before each gradient step
  irtkPublicAttributeMacro(bool, AdaptiveRemeshing);

protected:

  /// Common attributes of (untransformed) input target data sets
  ///
  /// These attributes are in particular used to initialize the control point
  /// grid of a free-form deformation such that the grid is large enough to
  /// be valid for every target point for which the transformation will be
  /// evaluated. It is therefore computed from the attributes of the input
  /// data sets rather than the downsampled data. If a _Domain mask is set,
  /// the attributes of this mask are copied instead.
  irtkImageAttributes _RegistrationDomain;

  vector<irtkPoint>                _Centroid;                       ///< Centroids of images
  irtkPoint                        _TargetOffset;                   ///< Target origin offset
  irtkPoint                        _SourceOffset;                   ///< Source origin offset
  vector<const irtkBaseImage *>    _Input;                          ///< Input images
  vector<ResampledImageList>       _Image;                          ///< Resolution pyramid
  vector<irtkBinaryImage *>        _Mask;                           ///< Domain on which to evaluate similarity
  irtkTransformation              *_Transformation;                 ///< Current estimate
  vector<TransformationInfo>       _TransformationInfo;             ///< Meta-data of partial transformation
  vector<irtkTransformation *>     _TransformationInstance;         ///< Partial transformations
  irtkRegistrationEnergy           _Energy;                         ///< Registration energy
  irtkLocalOptimizer              *_Optimizer;                      ///< Used optimizer
  irtkTransformationModel          _CurrentModel;                   ///< Current transformation model
  int                              _CurrentLevel;                   ///< Current resolution level
  irtkEventDelegate                _EventDelegate;                  ///< Forwards optimization events to observers
  string                           _EnergyFormula;                  ///< Registration energy formula as string
  vector<ImageSimilarityInfo>      _ImageSimilarityInfo;            ///< Parsed similarity measure(s)
  vector<ConstraintInfo>           _ConstraintInfo;                 ///< Parsed constraint(s)
  vector<irtkVector3D<double> >    _Resolution[MAX_NO_RESOLUTIONS]; ///< Image resolution in mm
  vector<double>                   _MinEdgeLength[MAX_NO_RESOLUTIONS]; ///< Minimum edge length in mm
  vector<double>                   _MaxEdgeLength[MAX_NO_RESOLUTIONS]; ///< Maximum edge length in mm
  int                              _UseGaussianResolutionPyramid;   ///< Whether resolution levels correspond to a Gaussian pyramid
  vector<double>                   _Blurring  [MAX_NO_RESOLUTIONS]; ///< Image blurring value
  vector<double>                   _Background;                     ///< Image background value
  vector<double>                   _Padding;                        ///< Image padding value
  bool                             _DownsampleWithPadding;          ///< Whether to take background into account
                                                                    ///< during initialization of the image pyramid
  bool                             _CropPadImages;                  ///< Whether to crop/pad input images
  int                              _CropPadFFD;                     ///< Whether to crop/pad FFD lattice

#ifdef HAS_VTK
  vector<double>                   _PointSetTime;                   ///< Time point of input points, curves, and/or surfaces
  vector<vtkSmartPointer<vtkPointSet> > _PointSetInput;             ///< Input points, curves, and/or surfaces
  vector<vector<vtkSmartPointer<vtkPointSet> > > _PointSet;         ///< Remeshed/-sampled points, curves, and/or surfaces
  vector<irtkRegisteredPointSet *> _PointSetOutput;                 ///< Output points, curves, and/or surfaces
  vector<PointSetOutputInfo>       _PointSetOutputInfo;             ///< Meta-data of output points, curves, and/or surfaces
  vector<PointSetDistanceInfo>     _PointSetDistanceInfo;           ///< Parsed polydata distance measure(s)
  vector<PointSetConstraintInfo>   _PointSetConstraintInfo;         ///< Parsed polydata constraint measure(s)
#else
  vector<double>                   _PointSetTime;                   ///< Unused
  vector<void *>                   _PointSetInput;                  ///< Unused
  vector<vector<void *> >          _PointSet;                       ///< Unused
  vector<void *>                   _PointSetOutput;                 ///< Unused
  vector<PointSetOutputInfo>       _PointSetOutputInfo;             ///< Unused
  vector<PointSetDistanceInfo>     _PointSetDistanceInfo;           ///< Unused
  vector<PointSetConstraintInfo>   _PointSetConstraintInfo;         ///< Unused
#endif

  int    _Centering             [MAX_NO_RESOLUTIONS];      ///< Whether to center foreground (if applicable)
  double _MinControlPointSpacing[MAX_NO_RESOLUTIONS][4];   ///< Control point spacing for FFDs
  double _MaxControlPointSpacing[MAX_NO_RESOLUTIONS][4];   ///< Control point spacing for FFDs
  bool   _Subdivide             [MAX_NO_RESOLUTIONS][4];   ///< Whether to subdivide FFD

  /// Parameters not considered by the registration filter itself
  ///
  /// These parameters are passed on to the sub-modules of the registration
  /// filter such as the image similarity measure(s), the registration
  /// constraint(s), and the optimizer. This way, the registration filter does
  /// not need to know which parameters its sub-modules accept and is better
  /// decoupled from the implementation of the respective image similarities,
  /// constraints, and optimizers.
  irtkParameterList _Parameter[MAX_NO_RESOLUTIONS];

  /// Delegate of pre-update function
  irtkRegistrationEnergy::PreUpdateFunctionType _PreUpdateDelegate;

  /// Info of cached displacement fields
  vector<DisplacementInfo>        _DisplacementInfo;
  vector<DisplacementImageType *> _DisplacementField;

  // ---------------------------------------------------------------------------
  // Access to managed data objects and their attributes

protected:

  /// Determine number of temporal frames and temporal sampling attributes
  virtual int NumberOfFrames(double * = NULL, double * = NULL, double * = NULL) const;

  /// Get attributes of n-th input image at specified resolution level
  virtual irtkImageAttributes ImageAttributes(int, int = -1) const;

  /// Common attributes of (untransformed) input target data sets at specified resolution level
  virtual irtkImageAttributes RegistrationDomain(int = -1) const;

  /// Average resolution of (untransformed) target data sets at given level
  virtual irtkVector3D<double> AverageOutputResolution(int = -1) const;

  /// Get (partial/inverse) output transformation
  virtual irtkTransformation *OutputTransformation(TransformationInfo);

  /// Initialize output image corresponding to registered (transformed) image
  virtual void SetInputOf(irtkRegisteredImage *, const irtkImageAttributes &, int, TransformationInfo);

#ifdef HAS_VTK
  /// Get (new) registered output point set
  virtual irtkRegisteredPointSet *OutputPointSet(int, double, TransformationInfo);
#endif

  // ---------------------------------------------------------------------------
  // Construction/Destruction
private:

  /// Copy constructor
  /// \note Intentionally not implemented
  irtkGenericRegistrationFilter(const irtkGenericRegistrationFilter &);

  /// Assignment operator
  /// \note Intentionally not implemented
  void operator =(const irtkGenericRegistrationFilter &);

public:

  /// Constructor
  irtkGenericRegistrationFilter();

  /// Reset filter settings, but keep input
  virtual void Reset();

  /// Reset filter settings and input
  virtual void Clear();

  /// Destructor
  virtual ~irtkGenericRegistrationFilter();

  // ---------------------------------------------------------------------------
  // Input images

  /// Set input images of the registration filter
  void Input(const irtkBaseImage *, const irtkBaseImage *);

  /// Set input images of the registration filter
  void Input(int, const irtkBaseImage **);

  /// Set input images of the registration filter
  template <class TVoxel> void Input(int, const irtkGenericImage<TVoxel> **);

  /// Add filter input image
  ///
  /// For registration filters which support multiple target/source images
  /// such as multiple channels and/or frames of a temporal image sequence.
  /// How the transformation is applied to which inputs depends on the
  /// respective transformation model and image registration method.
  void AddInput(const irtkBaseImage *);

  /// Number of input images
  int NumberOfImages() const;

  /// Number of required input images
  /// \note Only valid after ParseEnergyFormula has been called!
  int NumberOfRequiredImages() const;

  /// Determine whether the specified input image either remains untransformed,
  /// or is transformed by the inverse transformation or a part of it such
  /// as in case of an inverse consistent and/or symmetric energy function.
  bool IsTargetImage(int) const;

  /// Determine whether the specified input image will be transformed by the
  /// forward transformation or part of it such as in case of a symmetric energy.
  bool IsSourceImage(int) const;

  /// Determine whether the specified input image remains untransformed.
  bool IsFixedImage(int) const;

  /// Determine whether the specified input image will be transformed.
  bool IsMovingImage(int) const;

  // ---------------------------------------------------------------------------
  // Input simplicial complexes (points, curves, surfaces, tetrahedral meshes)
#ifdef HAS_VTK

  /// Set input point sets of the registration filter
  void Input(vtkPointSet *, vtkPointSet *, double = .0, double = 1.0);

  /// Set input point set of the registration filter
  void Input(int, vtkPointSet **, double * = NULL);

  /// Add filter input point set
  void AddInput(vtkPointSet *, double = .0);

#endif

  /// Number of input point sets
  int NumberOfPointSets() const;

  /// Number of required input point sets
  /// \note Only valid after ParseEnergyFormula has been called!
  int NumberOfRequiredPointSets() const;

  /// Determine whether the specified input point set will be transformed by the
  /// forward transformation or part of it such as in case of a symmetric energy.
  bool IsTargetPointSet(int) const;

  /// Determine whether the specified input point set either remains untransformed,
  /// or is transformed by the inverse transformation or a part of it such
  /// as in case of an inverse consistent and/or symmetric energy function.
  bool IsSourcePointSet(int) const;

  /// Determine whether the specified input point set remains untransformed.
  bool IsFixedPointSet(int) const;

  /// Determine whether the specified input point set will be transformed.
  bool IsMovingPointSet(int) const;

  // ---------------------------------------------------------------------------
  // Parameter

  using irtkRegistrationFilter::Read;

  /// Set (single) transformation model
  virtual void TransformationModel(irtkTransformationModel);

  /// Parse registration energy function
  /// \note Optional to call explicitly as this method is called by GuessParameter.
  virtual void ParseEnergyFormula(int = -1, int = -1, int = -1);

  /// Guess proper setting for any yet unset parameter
  /// \note Optional to call explicitly as this method is called by Run.
  virtual void GuessParameter();

  /// Read registration parameters from input stream
  virtual bool Read(istream &, bool = false);

  /// Set named parameter from value as string
  virtual bool Set(const char *, const char *);

  /// Set named parameter from value as string
  virtual bool Set(const char *, const char *, int);

  /// Get parameters as key/value as string map
  virtual irtkParameterList Parameter() const;

  /// Get parameters as key/value as string map
  virtual irtkParameterList Parameter(int) const;

  /// Write registration parameters to file
  virtual void Write(const char *) const;

  // ---------------------------------------------------------------------------
  // Execution

  /// Run the multi-level registration
  virtual void Run();

  // ---------------------------------------------------------------------------
  // Implementation - the following functions can be overridden in subclasses

protected:

  /// Whether current level is the initial resolution level
  bool InitialLevel() const;

  /// Whether current level is the final resolution level
  bool FinalLevel() const;

  /// Run multi-resolution registration
  virtual void MultiResolutionOptimization();

  /// Initialize registration at current resolution
  virtual void Initialize();

  /// Initialize image resolution pyramid
  virtual void InitializePyramid();
  void InitializePyramid_v20_or_v21(); // emulates original registration2++ implementation
  void InitializePyramid_v22();        // emulates current  registration2++ implementation

  /// Remesh/-sample input point sets
  virtual void InitializePointSets();

  /// Type of output transformation of sub-registration at current resolution
  virtual irtkTransformationType TransformationType();

  /// Initialize new transformation instance
  virtual void InitializeTransformation();

  /// Make an initial guess of the (global) output transformation
  virtual irtkTransformation *MakeInitialGuess();

  /// Initialize transformation parameters using provided initial guess
  virtual void ApplyInitialGuess();

  /// Initialize status of linear parameters
  virtual void InitializeStatus(irtkHomogeneousTransformation *);

  /// Initialize status of FFD parameters
  virtual void InitializeStatus(irtkFreeFormTransformation *);

  /// Initialize status of transformation parameters
  virtual void InitializeStatus();

  /// Initialize transformation for sub-registration at current resolution
  virtual void InitializeOutput();

  /// Instantiate new image similarity term for energy function
  /// \note The individual energy terms are destroyed by the energy function!
  virtual void AddImageSimilarityTerm();

  /// Instantiate new point set distance term for energy function
  /// \note The individual energy terms are destroyed by the energy function!
  virtual void AddPointSetDistanceTerm();

  /// Instantiate new point set constraint term for energy function
  /// \note The individual energy terms are destroyed by the energy function!
  virtual void AddPointSetConstraintTerm();

  /// Instantiate new regularization term for energy function
  /// \note The individual energy terms are destroyed by the energy function!
  virtual void AddPenaltyTerm();

  /// Initialize registration energy of registration at current resolution
  virtual void InitializeEnergy();

  /// Initialize optimizer used to solve registration problem
  virtual void InitializeOptimizer();

  /// Finalize registration at current resolution
  virtual void Finalize();

  /// Callback function called by _Energy->Update(bool)
  void PreUpdateCallback(bool);

};

////////////////////////////////////////////////////////////////////////////////
// Inline/template definitions
////////////////////////////////////////////////////////////////////////////////

// -----------------------------------------------------------------------------
template <class TVoxel>
void irtkGenericRegistrationFilter::Input(int num, const irtkGenericImage<TVoxel> **image)
{
  _Input.clear();
  for (int n = 0; n < num; ++n) AddInput(image[n]);
}


#endif
