��	
��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
�
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718��
�
!embedding_layer_origin/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
�(�*2
shared_name#!embedding_layer_origin/embeddings
�
5embedding_layer_origin/embeddings/Read/ReadVariableOpReadVariableOp!embedding_layer_origin/embeddings* 
_output_shapes
:
�(�*
dtype0
�
!embedding_layer_target/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*2
shared_name#!embedding_layer_target/embeddings
�
5embedding_layer_target/embeddings/Read/ReadVariableOpReadVariableOp!embedding_layer_target/embeddings* 
_output_shapes
:
��*
dtype0
�
fully_connected_source/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*.
shared_namefully_connected_source/kernel
�
1fully_connected_source/kernel/Read/ReadVariableOpReadVariableOpfully_connected_source/kernel* 
_output_shapes
:
��*
dtype0
�
fully_connected_source/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namefully_connected_source/bias
�
/fully_connected_source/bias/Read/ReadVariableOpReadVariableOpfully_connected_source/bias*
_output_shapes	
:�*
dtype0
�
fully_connected_target/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*.
shared_namefully_connected_target/kernel
�
1fully_connected_target/kernel/Read/ReadVariableOpReadVariableOpfully_connected_target/kernel* 
_output_shapes
:
��*
dtype0
�
fully_connected_target/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namefully_connected_target/bias
�
/fully_connected_target/bias/Read/ReadVariableOpReadVariableOpfully_connected_target/bias*
_output_shapes	
:�*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
��*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:�*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
��*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:�*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

NoOpNoOp
�+
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�*
value�*B�* B�*
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer-11
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
 
 
b

embeddings
regularization_losses
trainable_variables
	variables
	keras_api
b

embeddings
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
 	keras_api
R
!regularization_losses
"trainable_variables
#	variables
$	keras_api
h

%kernel
&bias
'regularization_losses
(trainable_variables
)	variables
*	keras_api
h

+kernel
,bias
-regularization_losses
.trainable_variables
/	variables
0	keras_api
R
1regularization_losses
2trainable_variables
3	variables
4	keras_api
h

5kernel
6bias
7regularization_losses
8trainable_variables
9	variables
:	keras_api
h

;kernel
<bias
=regularization_losses
>trainable_variables
?	variables
@	keras_api
R
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
 
 
F
0
1
%2
&3
+4
,5
56
67
;8
<9
F
0
1
%2
&3
+4
,5
56
67
;8
<9
�
Elayer_regularization_losses
Fmetrics
Glayer_metrics
regularization_losses
Hnon_trainable_variables

Ilayers
trainable_variables
	variables
 
qo
VARIABLE_VALUE!embedding_layer_origin/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
�
Jlayer_regularization_losses
Kmetrics
Llayer_metrics
regularization_losses
Mnon_trainable_variables

Nlayers
trainable_variables
	variables
qo
VARIABLE_VALUE!embedding_layer_target/embeddings:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
�
Olayer_regularization_losses
Pmetrics
Qlayer_metrics
regularization_losses
Rnon_trainable_variables

Slayers
trainable_variables
	variables
 
 
 
�
Tlayer_regularization_losses
Umetrics
Vlayer_metrics
regularization_losses
Wnon_trainable_variables

Xlayers
trainable_variables
	variables
 
 
 
�
Ylayer_regularization_losses
Zmetrics
[layer_metrics
!regularization_losses
\non_trainable_variables

]layers
"trainable_variables
#	variables
ig
VARIABLE_VALUEfully_connected_source/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEfully_connected_source/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

%0
&1

%0
&1
�
^layer_regularization_losses
_metrics
`layer_metrics
'regularization_losses
anon_trainable_variables

blayers
(trainable_variables
)	variables
ig
VARIABLE_VALUEfully_connected_target/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEfully_connected_target/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

+0
,1

+0
,1
�
clayer_regularization_losses
dmetrics
elayer_metrics
-regularization_losses
fnon_trainable_variables

glayers
.trainable_variables
/	variables
 
 
 
�
hlayer_regularization_losses
imetrics
jlayer_metrics
1regularization_losses
knon_trainable_variables

llayers
2trainable_variables
3	variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

50
61

50
61
�
mlayer_regularization_losses
nmetrics
olayer_metrics
7regularization_losses
pnon_trainable_variables

qlayers
8trainable_variables
9	variables
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

;0
<1

;0
<1
�
rlayer_regularization_losses
smetrics
tlayer_metrics
=regularization_losses
unon_trainable_variables

vlayers
>trainable_variables
?	variables
 
 
 
�
wlayer_regularization_losses
xmetrics
ylayer_metrics
Aregularization_losses
znon_trainable_variables

{layers
Btrainable_variables
C	variables
 

|0
}1
 
 
V
0
1
2
3
4
5
6
7
	8

9
10
11
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
6
	~total
	count
�	variables
�	keras_api
I

�total

�count
�
_fn_kwargs
�	variables
�	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

~0
1

�	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�	variables

serving_default_input_sourcePlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������

serving_default_input_targetPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_sourceserving_default_input_target!embedding_layer_target/embeddings!embedding_layer_origin/embeddingsfully_connected_source/kernelfully_connected_source/biasfully_connected_target/kernelfully_connected_target/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference_signature_wrapper_674
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename5embedding_layer_origin/embeddings/Read/ReadVariableOp5embedding_layer_target/embeddings/Read/ReadVariableOp1fully_connected_source/kernel/Read/ReadVariableOp/fully_connected_source/bias/Read/ReadVariableOp1fully_connected_target/kernel/Read/ReadVariableOp/fully_connected_target/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *&
f!R
__inference__traced_save_1052
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename!embedding_layer_origin/embeddings!embedding_layer_target/embeddingsfully_connected_source/kernelfully_connected_source/biasfully_connected_target/kernelfully_connected_target/biasdense/kernel
dense/biasdense_1/kerneldense_1/biastotalcounttotal_1count_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *)
f$R"
 __inference__traced_restore_1104��
�J
�	
>__inference_model_layer_call_and_return_conditional_losses_828
inputs_0
inputs_1?
+embedding_layer_target_embedding_lookup_782:
��?
+embedding_layer_origin_embedding_lookup_788:
�(�I
5fully_connected_source_matmul_readvariableop_resource:
��E
6fully_connected_source_biasadd_readvariableop_resource:	�I
5fully_connected_target_matmul_readvariableop_resource:
��E
6fully_connected_target_biasadd_readvariableop_resource:	�8
$dense_matmul_readvariableop_resource:
��4
%dense_biasadd_readvariableop_resource:	�:
&dense_1_matmul_readvariableop_resource:
��6
'dense_1_biasadd_readvariableop_resource:	�
identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�'embedding_layer_origin/embedding_lookup�'embedding_layer_target/embedding_lookup�-fully_connected_source/BiasAdd/ReadVariableOp�,fully_connected_source/MatMul/ReadVariableOp�-fully_connected_target/BiasAdd/ReadVariableOp�,fully_connected_target/MatMul/ReadVariableOp�
embedding_layer_target/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:���������2
embedding_layer_target/Cast�
'embedding_layer_target/embedding_lookupResourceGather+embedding_layer_target_embedding_lookup_782embedding_layer_target/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*>
_class4
20loc:@embedding_layer_target/embedding_lookup/782*,
_output_shapes
:����������*
dtype02)
'embedding_layer_target/embedding_lookup�
0embedding_layer_target/embedding_lookup/IdentityIdentity0embedding_layer_target/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*>
_class4
20loc:@embedding_layer_target/embedding_lookup/782*,
_output_shapes
:����������22
0embedding_layer_target/embedding_lookup/Identity�
2embedding_layer_target/embedding_lookup/Identity_1Identity9embedding_layer_target/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:����������24
2embedding_layer_target/embedding_lookup/Identity_1�
embedding_layer_origin/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:���������2
embedding_layer_origin/Cast�
'embedding_layer_origin/embedding_lookupResourceGather+embedding_layer_origin_embedding_lookup_788embedding_layer_origin/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*>
_class4
20loc:@embedding_layer_origin/embedding_lookup/788*,
_output_shapes
:����������*
dtype02)
'embedding_layer_origin/embedding_lookup�
0embedding_layer_origin/embedding_lookup/IdentityIdentity0embedding_layer_origin/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*>
_class4
20loc:@embedding_layer_origin/embedding_lookup/788*,
_output_shapes
:����������22
0embedding_layer_origin/embedding_lookup/Identity�
2embedding_layer_origin/embedding_lookup/Identity_1Identity9embedding_layer_origin/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:����������24
2embedding_layer_origin/embedding_lookup/Identity_1s
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
flatten_1/Const�
flatten_1/ReshapeReshape;embedding_layer_target/embedding_lookup/Identity_1:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_1/Reshapeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����X  2
flatten/Const�
flatten/ReshapeReshape;embedding_layer_origin/embedding_lookup/Identity_1:output:0flatten/Const:output:0*
T0*(
_output_shapes
:����������2
flatten/Reshape�
,fully_connected_source/MatMul/ReadVariableOpReadVariableOp5fully_connected_source_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02.
,fully_connected_source/MatMul/ReadVariableOp�
fully_connected_source/MatMulMatMulflatten/Reshape:output:04fully_connected_source/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
fully_connected_source/MatMul�
-fully_connected_source/BiasAdd/ReadVariableOpReadVariableOp6fully_connected_source_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-fully_connected_source/BiasAdd/ReadVariableOp�
fully_connected_source/BiasAddBiasAdd'fully_connected_source/MatMul:product:05fully_connected_source/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2 
fully_connected_source/BiasAdd�
fully_connected_source/ReluRelu'fully_connected_source/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
fully_connected_source/Relu�
,fully_connected_target/MatMul/ReadVariableOpReadVariableOp5fully_connected_target_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02.
,fully_connected_target/MatMul/ReadVariableOp�
fully_connected_target/MatMulMatMulflatten_1/Reshape:output:04fully_connected_target/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
fully_connected_target/MatMul�
-fully_connected_target/BiasAdd/ReadVariableOpReadVariableOp6fully_connected_target_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-fully_connected_target/BiasAdd/ReadVariableOp�
fully_connected_target/BiasAddBiasAdd'fully_connected_target/MatMul:product:05fully_connected_target/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2 
fully_connected_target/BiasAdd�
fully_connected_target/ReluRelu'fully_connected_target/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
fully_connected_target/Relut
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis�
concatenate/concatConcatV2)fully_connected_source/Relu:activations:0)fully_connected_target/Relu:activations:0 concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������2
concatenate/concat�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMulconcatenate/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:����������2

dense/Relu�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_1/BiasAddz
softmax/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
softmax/Softmax�
IdentityIdentitysoftmax/Softmax:softmax:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp(^embedding_layer_origin/embedding_lookup(^embedding_layer_target/embedding_lookup.^fully_connected_source/BiasAdd/ReadVariableOp-^fully_connected_source/MatMul/ReadVariableOp.^fully_connected_target/BiasAdd/ReadVariableOp-^fully_connected_target/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::���������:���������: : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2R
'embedding_layer_origin/embedding_lookup'embedding_layer_origin/embedding_lookup2R
'embedding_layer_target/embedding_lookup'embedding_layer_target/embedding_lookup2^
-fully_connected_source/BiasAdd/ReadVariableOp-fully_connected_source/BiasAdd/ReadVariableOp2\
,fully_connected_source/MatMul/ReadVariableOp,fully_connected_source/MatMul/ReadVariableOp2^
-fully_connected_target/BiasAdd/ReadVariableOp-fully_connected_target/BiasAdd/ReadVariableOp2\
,fully_connected_target/MatMul/ReadVariableOp,fully_connected_target/MatMul/ReadVariableOp:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
A
%__inference_softmax_layer_call_fn_981

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_softmax_layer_call_and_return_conditional_losses_3562
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
4__inference_fully_connected_source_layer_call_fn_893

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_fully_connected_source_layer_call_and_return_conditional_losses_2862
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
O__inference_fully_connected_source_layer_call_and_return_conditional_losses_286

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
\
@__inference_flatten_layer_call_and_return_conditional_losses_273

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����X  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
@__inference_dense_1_layer_call_and_return_conditional_losses_976

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
#__inference_model_layer_call_fn_382
input_source
input_target
unknown:
��
	unknown_0:
�(�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_sourceinput_targetunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_model_layer_call_and_return_conditional_losses_3592
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::���������:���������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
'
_output_shapes
:���������
&
_user_specified_nameinput_source:UQ
'
_output_shapes
:���������
&
_user_specified_nameinput_target
�?
�
 __inference__traced_restore_1104
file_prefixF
2assignvariableop_embedding_layer_origin_embeddings:
�(�H
4assignvariableop_1_embedding_layer_target_embeddings:
��D
0assignvariableop_2_fully_connected_source_kernel:
��=
.assignvariableop_3_fully_connected_source_bias:	�D
0assignvariableop_4_fully_connected_target_kernel:
��=
.assignvariableop_5_fully_connected_target_bias:	�3
assignvariableop_6_dense_kernel:
��,
assignvariableop_7_dense_bias:	�5
!assignvariableop_8_dense_1_kernel:
��.
assignvariableop_9_dense_1_bias:	�#
assignvariableop_10_total: #
assignvariableop_11_count: %
assignvariableop_12_total_1: %
assignvariableop_13_count_1: 
identity_15��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*P
_output_shapes>
<:::::::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp2assignvariableop_embedding_layer_origin_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp4assignvariableop_1_embedding_layer_target_embeddingsIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp0assignvariableop_2_fully_connected_source_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp.assignvariableop_3_fully_connected_source_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp0assignvariableop_4_fully_connected_target_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp.assignvariableop_5_fully_connected_target_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOpassignvariableop_12_total_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOpassignvariableop_13_count_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_139
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_14Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_14�
Identity_15IdentityIdentity_14:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_15"#
identity_15Identity_15:output:0*1
_input_shapes 
: : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
#__inference_model_layer_call_fn_576
input_source
input_target
unknown:
��
	unknown_0:
�(�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_sourceinput_targetunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_model_layer_call_and_return_conditional_losses_5272
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::���������:���������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
'
_output_shapes
:���������
&
_user_specified_nameinput_source:UQ
'
_output_shapes
:���������
&
_user_specified_nameinput_target
�1
�
>__inference_model_layer_call_and_return_conditional_losses_611
input_source
input_target.
embedding_layer_target_580:
��.
embedding_layer_origin_583:
�(�.
fully_connected_source_588:
��)
fully_connected_source_590:	�.
fully_connected_target_593:
��)
fully_connected_target_595:	�
	dense_599:
��
	dense_601:	�
dense_1_604:
��
dense_1_606:	�
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�.embedding_layer_origin/StatefulPartitionedCall�.embedding_layer_target/StatefulPartitionedCall�.fully_connected_source/StatefulPartitionedCall�.fully_connected_target/StatefulPartitionedCall�
.embedding_layer_target/StatefulPartitionedCallStatefulPartitionedCallinput_targetembedding_layer_target_580*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_embedding_layer_target_layer_call_and_return_conditional_losses_24120
.embedding_layer_target/StatefulPartitionedCall�
.embedding_layer_origin/StatefulPartitionedCallStatefulPartitionedCallinput_sourceembedding_layer_origin_583*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_embedding_layer_origin_layer_call_and_return_conditional_losses_25520
.embedding_layer_origin/StatefulPartitionedCall�
flatten_1/PartitionedCallPartitionedCall7embedding_layer_target/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_flatten_1_layer_call_and_return_conditional_losses_2652
flatten_1/PartitionedCall�
flatten/PartitionedCallPartitionedCall7embedding_layer_origin/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_2732
flatten/PartitionedCall�
.fully_connected_source/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0fully_connected_source_588fully_connected_source_590*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_fully_connected_source_layer_call_and_return_conditional_losses_28620
.fully_connected_source/StatefulPartitionedCall�
.fully_connected_target/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0fully_connected_target_593fully_connected_target_595*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_fully_connected_target_layer_call_and_return_conditional_losses_30320
.fully_connected_target/StatefulPartitionedCall�
concatenate/PartitionedCallPartitionedCall7fully_connected_source/StatefulPartitionedCall:output:07fully_connected_target/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_concatenate_layer_call_and_return_conditional_losses_3162
concatenate/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0	dense_599	dense_601*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_3292
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_604dense_1_606*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_3452!
dense_1/StatefulPartitionedCall�
softmax/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_softmax_layer_call_and_return_conditional_losses_3562
softmax/PartitionedCall�
IdentityIdentity softmax/PartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall/^embedding_layer_origin/StatefulPartitionedCall/^embedding_layer_target/StatefulPartitionedCall/^fully_connected_source/StatefulPartitionedCall/^fully_connected_target/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::���������:���������: : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2`
.embedding_layer_origin/StatefulPartitionedCall.embedding_layer_origin/StatefulPartitionedCall2`
.embedding_layer_target/StatefulPartitionedCall.embedding_layer_target/StatefulPartitionedCall2`
.fully_connected_source/StatefulPartitionedCall.fully_connected_source/StatefulPartitionedCall2`
.fully_connected_target/StatefulPartitionedCall.fully_connected_target/StatefulPartitionedCall:U Q
'
_output_shapes
:���������
&
_user_specified_nameinput_source:UQ
'
_output_shapes
:���������
&
_user_specified_nameinput_target
�
U
)__inference_concatenate_layer_call_fn_930
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_concatenate_layer_call_and_return_conditional_losses_3162
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������:����������:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/1
�

�
!__inference_signature_wrapper_674
input_source
input_target
unknown:
��
	unknown_0:
�(�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_sourceinput_targetunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *'
f"R 
__inference__wrapped_model_2222
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::���������:���������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
'
_output_shapes
:���������
&
_user_specified_nameinput_source:UQ
'
_output_shapes
:���������
&
_user_specified_nameinput_target
�
�
4__inference_embedding_layer_origin_layer_call_fn_835

inputs
unknown:
�(�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_embedding_layer_origin_layer_call_and_return_conditional_losses_2552
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�1
�
>__inference_model_layer_call_and_return_conditional_losses_646
input_source
input_target.
embedding_layer_target_615:
��.
embedding_layer_origin_618:
�(�.
fully_connected_source_623:
��)
fully_connected_source_625:	�.
fully_connected_target_628:
��)
fully_connected_target_630:	�
	dense_634:
��
	dense_636:	�
dense_1_639:
��
dense_1_641:	�
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�.embedding_layer_origin/StatefulPartitionedCall�.embedding_layer_target/StatefulPartitionedCall�.fully_connected_source/StatefulPartitionedCall�.fully_connected_target/StatefulPartitionedCall�
.embedding_layer_target/StatefulPartitionedCallStatefulPartitionedCallinput_targetembedding_layer_target_615*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_embedding_layer_target_layer_call_and_return_conditional_losses_24120
.embedding_layer_target/StatefulPartitionedCall�
.embedding_layer_origin/StatefulPartitionedCallStatefulPartitionedCallinput_sourceembedding_layer_origin_618*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_embedding_layer_origin_layer_call_and_return_conditional_losses_25520
.embedding_layer_origin/StatefulPartitionedCall�
flatten_1/PartitionedCallPartitionedCall7embedding_layer_target/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_flatten_1_layer_call_and_return_conditional_losses_2652
flatten_1/PartitionedCall�
flatten/PartitionedCallPartitionedCall7embedding_layer_origin/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_2732
flatten/PartitionedCall�
.fully_connected_source/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0fully_connected_source_623fully_connected_source_625*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_fully_connected_source_layer_call_and_return_conditional_losses_28620
.fully_connected_source/StatefulPartitionedCall�
.fully_connected_target/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0fully_connected_target_628fully_connected_target_630*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_fully_connected_target_layer_call_and_return_conditional_losses_30320
.fully_connected_target/StatefulPartitionedCall�
concatenate/PartitionedCallPartitionedCall7fully_connected_source/StatefulPartitionedCall:output:07fully_connected_target/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_concatenate_layer_call_and_return_conditional_losses_3162
concatenate/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0	dense_634	dense_636*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_3292
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_639dense_1_641*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_3452!
dense_1/StatefulPartitionedCall�
softmax/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_softmax_layer_call_and_return_conditional_losses_3562
softmax/PartitionedCall�
IdentityIdentity softmax/PartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall/^embedding_layer_origin/StatefulPartitionedCall/^embedding_layer_target/StatefulPartitionedCall/^fully_connected_source/StatefulPartitionedCall/^fully_connected_target/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::���������:���������: : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2`
.embedding_layer_origin/StatefulPartitionedCall.embedding_layer_origin/StatefulPartitionedCall2`
.embedding_layer_target/StatefulPartitionedCall.embedding_layer_target/StatefulPartitionedCall2`
.fully_connected_source/StatefulPartitionedCall.fully_connected_source/StatefulPartitionedCall2`
.fully_connected_target/StatefulPartitionedCall.fully_connected_target/StatefulPartitionedCall:U Q
'
_output_shapes
:���������
&
_user_specified_nameinput_source:UQ
'
_output_shapes
:���������
&
_user_specified_nameinput_target
�
A
%__inference_flatten_layer_call_fn_867

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_2732
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
4__inference_fully_connected_target_layer_call_fn_913

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_fully_connected_target_layer_call_and_return_conditional_losses_3032
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
O__inference_embedding_layer_target_layer_call_and_return_conditional_losses_862

inputs(
embedding_lookup_856:
��
identity��embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:���������2
Cast�
embedding_lookupResourceGatherembedding_lookup_856Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*'
_class
loc:@embedding_lookup/856*,
_output_shapes
:����������*
dtype02
embedding_lookup�
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@embedding_lookup/856*,
_output_shapes
:����������2
embedding_lookup/Identity�
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:����������2
embedding_lookup/Identity_1�
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
#__inference_dense_layer_call_fn_946

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_3292
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
4__inference_embedding_layer_target_layer_call_fn_852

inputs
unknown:
��
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_embedding_layer_target_layer_call_and_return_conditional_losses_2412
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
^
B__inference_flatten_1_layer_call_and_return_conditional_losses_265

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
O__inference_embedding_layer_origin_layer_call_and_return_conditional_losses_845

inputs(
embedding_lookup_839:
�(�
identity��embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:���������2
Cast�
embedding_lookupResourceGatherembedding_lookup_839Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*'
_class
loc:@embedding_lookup/839*,
_output_shapes
:����������*
dtype02
embedding_lookup�
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@embedding_lookup/839*,
_output_shapes
:����������2
embedding_lookup/Identity�
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:����������2
embedding_lookup/Identity_1�
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
\
@__inference_flatten_layer_call_and_return_conditional_losses_873

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����X  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�1
�
>__inference_model_layer_call_and_return_conditional_losses_359

inputs
inputs_1.
embedding_layer_target_242:
��.
embedding_layer_origin_256:
�(�.
fully_connected_source_287:
��)
fully_connected_source_289:	�.
fully_connected_target_304:
��)
fully_connected_target_306:	�
	dense_330:
��
	dense_332:	�
dense_1_346:
��
dense_1_348:	�
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�.embedding_layer_origin/StatefulPartitionedCall�.embedding_layer_target/StatefulPartitionedCall�.fully_connected_source/StatefulPartitionedCall�.fully_connected_target/StatefulPartitionedCall�
.embedding_layer_target/StatefulPartitionedCallStatefulPartitionedCallinputs_1embedding_layer_target_242*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_embedding_layer_target_layer_call_and_return_conditional_losses_24120
.embedding_layer_target/StatefulPartitionedCall�
.embedding_layer_origin/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_layer_origin_256*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_embedding_layer_origin_layer_call_and_return_conditional_losses_25520
.embedding_layer_origin/StatefulPartitionedCall�
flatten_1/PartitionedCallPartitionedCall7embedding_layer_target/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_flatten_1_layer_call_and_return_conditional_losses_2652
flatten_1/PartitionedCall�
flatten/PartitionedCallPartitionedCall7embedding_layer_origin/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_2732
flatten/PartitionedCall�
.fully_connected_source/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0fully_connected_source_287fully_connected_source_289*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_fully_connected_source_layer_call_and_return_conditional_losses_28620
.fully_connected_source/StatefulPartitionedCall�
.fully_connected_target/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0fully_connected_target_304fully_connected_target_306*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_fully_connected_target_layer_call_and_return_conditional_losses_30320
.fully_connected_target/StatefulPartitionedCall�
concatenate/PartitionedCallPartitionedCall7fully_connected_source/StatefulPartitionedCall:output:07fully_connected_target/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_concatenate_layer_call_and_return_conditional_losses_3162
concatenate/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0	dense_330	dense_332*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_3292
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_346dense_1_348*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_3452!
dense_1/StatefulPartitionedCall�
softmax/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_softmax_layer_call_and_return_conditional_losses_3562
softmax/PartitionedCall�
IdentityIdentity softmax/PartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall/^embedding_layer_origin/StatefulPartitionedCall/^embedding_layer_target/StatefulPartitionedCall/^fully_connected_source/StatefulPartitionedCall/^fully_connected_target/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::���������:���������: : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2`
.embedding_layer_origin/StatefulPartitionedCall.embedding_layer_origin/StatefulPartitionedCall2`
.embedding_layer_target/StatefulPartitionedCall.embedding_layer_target/StatefulPartitionedCall2`
.fully_connected_source/StatefulPartitionedCall.fully_connected_source/StatefulPartitionedCall2`
.fully_connected_target/StatefulPartitionedCall.fully_connected_target/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
>__inference_dense_layer_call_and_return_conditional_losses_329

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
O__inference_fully_connected_target_layer_call_and_return_conditional_losses_924

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
%__inference_dense_1_layer_call_fn_966

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_3452
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
>__inference_dense_layer_call_and_return_conditional_losses_957

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
n
D__inference_concatenate_layer_call_and_return_conditional_losses_316

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:����������2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:PL
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
O__inference_fully_connected_target_layer_call_and_return_conditional_losses_303

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
\
@__inference_softmax_layer_call_and_return_conditional_losses_986

inputs
identityX
SoftmaxSoftmaxinputs*
T0*(
_output_shapes
:����������2	
Softmaxf
IdentityIdentitySoftmax:softmax:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
O__inference_embedding_layer_origin_layer_call_and_return_conditional_losses_255

inputs(
embedding_lookup_249:
�(�
identity��embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:���������2
Cast�
embedding_lookupResourceGatherembedding_lookup_249Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*'
_class
loc:@embedding_lookup/249*,
_output_shapes
:����������*
dtype02
embedding_lookup�
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@embedding_lookup/249*,
_output_shapes
:����������2
embedding_lookup/Identity�
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:����������2
embedding_lookup/Identity_1�
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�J
�	
>__inference_model_layer_call_and_return_conditional_losses_777
inputs_0
inputs_1?
+embedding_layer_target_embedding_lookup_731:
��?
+embedding_layer_origin_embedding_lookup_737:
�(�I
5fully_connected_source_matmul_readvariableop_resource:
��E
6fully_connected_source_biasadd_readvariableop_resource:	�I
5fully_connected_target_matmul_readvariableop_resource:
��E
6fully_connected_target_biasadd_readvariableop_resource:	�8
$dense_matmul_readvariableop_resource:
��4
%dense_biasadd_readvariableop_resource:	�:
&dense_1_matmul_readvariableop_resource:
��6
'dense_1_biasadd_readvariableop_resource:	�
identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�'embedding_layer_origin/embedding_lookup�'embedding_layer_target/embedding_lookup�-fully_connected_source/BiasAdd/ReadVariableOp�,fully_connected_source/MatMul/ReadVariableOp�-fully_connected_target/BiasAdd/ReadVariableOp�,fully_connected_target/MatMul/ReadVariableOp�
embedding_layer_target/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:���������2
embedding_layer_target/Cast�
'embedding_layer_target/embedding_lookupResourceGather+embedding_layer_target_embedding_lookup_731embedding_layer_target/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*>
_class4
20loc:@embedding_layer_target/embedding_lookup/731*,
_output_shapes
:����������*
dtype02)
'embedding_layer_target/embedding_lookup�
0embedding_layer_target/embedding_lookup/IdentityIdentity0embedding_layer_target/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*>
_class4
20loc:@embedding_layer_target/embedding_lookup/731*,
_output_shapes
:����������22
0embedding_layer_target/embedding_lookup/Identity�
2embedding_layer_target/embedding_lookup/Identity_1Identity9embedding_layer_target/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:����������24
2embedding_layer_target/embedding_lookup/Identity_1�
embedding_layer_origin/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:���������2
embedding_layer_origin/Cast�
'embedding_layer_origin/embedding_lookupResourceGather+embedding_layer_origin_embedding_lookup_737embedding_layer_origin/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*>
_class4
20loc:@embedding_layer_origin/embedding_lookup/737*,
_output_shapes
:����������*
dtype02)
'embedding_layer_origin/embedding_lookup�
0embedding_layer_origin/embedding_lookup/IdentityIdentity0embedding_layer_origin/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*>
_class4
20loc:@embedding_layer_origin/embedding_lookup/737*,
_output_shapes
:����������22
0embedding_layer_origin/embedding_lookup/Identity�
2embedding_layer_origin/embedding_lookup/Identity_1Identity9embedding_layer_origin/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:����������24
2embedding_layer_origin/embedding_lookup/Identity_1s
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
flatten_1/Const�
flatten_1/ReshapeReshape;embedding_layer_target/embedding_lookup/Identity_1:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_1/Reshapeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����X  2
flatten/Const�
flatten/ReshapeReshape;embedding_layer_origin/embedding_lookup/Identity_1:output:0flatten/Const:output:0*
T0*(
_output_shapes
:����������2
flatten/Reshape�
,fully_connected_source/MatMul/ReadVariableOpReadVariableOp5fully_connected_source_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02.
,fully_connected_source/MatMul/ReadVariableOp�
fully_connected_source/MatMulMatMulflatten/Reshape:output:04fully_connected_source/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
fully_connected_source/MatMul�
-fully_connected_source/BiasAdd/ReadVariableOpReadVariableOp6fully_connected_source_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-fully_connected_source/BiasAdd/ReadVariableOp�
fully_connected_source/BiasAddBiasAdd'fully_connected_source/MatMul:product:05fully_connected_source/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2 
fully_connected_source/BiasAdd�
fully_connected_source/ReluRelu'fully_connected_source/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
fully_connected_source/Relu�
,fully_connected_target/MatMul/ReadVariableOpReadVariableOp5fully_connected_target_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02.
,fully_connected_target/MatMul/ReadVariableOp�
fully_connected_target/MatMulMatMulflatten_1/Reshape:output:04fully_connected_target/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
fully_connected_target/MatMul�
-fully_connected_target/BiasAdd/ReadVariableOpReadVariableOp6fully_connected_target_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-fully_connected_target/BiasAdd/ReadVariableOp�
fully_connected_target/BiasAddBiasAdd'fully_connected_target/MatMul:product:05fully_connected_target/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2 
fully_connected_target/BiasAdd�
fully_connected_target/ReluRelu'fully_connected_target/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
fully_connected_target/Relut
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis�
concatenate/concatConcatV2)fully_connected_source/Relu:activations:0)fully_connected_target/Relu:activations:0 concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������2
concatenate/concat�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMulconcatenate/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:����������2

dense/Relu�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_1/BiasAddz
softmax/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
softmax/Softmax�
IdentityIdentitysoftmax/Softmax:softmax:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp(^embedding_layer_origin/embedding_lookup(^embedding_layer_target/embedding_lookup.^fully_connected_source/BiasAdd/ReadVariableOp-^fully_connected_source/MatMul/ReadVariableOp.^fully_connected_target/BiasAdd/ReadVariableOp-^fully_connected_target/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::���������:���������: : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2R
'embedding_layer_origin/embedding_lookup'embedding_layer_origin/embedding_lookup2R
'embedding_layer_target/embedding_lookup'embedding_layer_target/embedding_lookup2^
-fully_connected_source/BiasAdd/ReadVariableOp-fully_connected_source/BiasAdd/ReadVariableOp2\
,fully_connected_source/MatMul/ReadVariableOp,fully_connected_source/MatMul/ReadVariableOp2^
-fully_connected_target/BiasAdd/ReadVariableOp-fully_connected_target/BiasAdd/ReadVariableOp2\
,fully_connected_target/MatMul/ReadVariableOp,fully_connected_target/MatMul/ReadVariableOp:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�

�
#__inference_model_layer_call_fn_700
inputs_0
inputs_1
unknown:
��
	unknown_0:
�(�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_model_layer_call_and_return_conditional_losses_3592
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::���������:���������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
C
'__inference_flatten_1_layer_call_fn_878

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_flatten_1_layer_call_and_return_conditional_losses_2652
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
O__inference_fully_connected_source_layer_call_and_return_conditional_losses_904

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
^
B__inference_flatten_1_layer_call_and_return_conditional_losses_884

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
#__inference_model_layer_call_fn_726
inputs_0
inputs_1
unknown:
��
	unknown_0:
�(�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_model_layer_call_and_return_conditional_losses_5272
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::���������:���������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�(
�
__inference__traced_save_1052
file_prefix@
<savev2_embedding_layer_origin_embeddings_read_readvariableop@
<savev2_embedding_layer_target_embeddings_read_readvariableop<
8savev2_fully_connected_source_kernel_read_readvariableop:
6savev2_fully_connected_source_bias_read_readvariableop<
8savev2_fully_connected_target_kernel_read_readvariableop:
6savev2_fully_connected_target_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0<savev2_embedding_layer_origin_embeddings_read_readvariableop<savev2_embedding_layer_target_embeddings_read_readvariableop8savev2_fully_connected_source_kernel_read_readvariableop6savev2_fully_connected_source_bias_read_readvariableop8savev2_fully_connected_target_kernel_read_readvariableop6savev2_fully_connected_target_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapesr
p: :
�(�:
��:
��:�:
��:�:
��:�:
��:�: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
�(�:&"
 
_output_shapes
:
��:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&	"
 
_output_shapes
:
��:!


_output_shapes	
:�:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
\
@__inference_softmax_layer_call_and_return_conditional_losses_356

inputs
identityX
SoftmaxSoftmaxinputs*
T0*(
_output_shapes
:����������2	
Softmaxf
IdentityIdentitySoftmax:softmax:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�1
�
>__inference_model_layer_call_and_return_conditional_losses_527

inputs
inputs_1.
embedding_layer_target_496:
��.
embedding_layer_origin_499:
�(�.
fully_connected_source_504:
��)
fully_connected_source_506:	�.
fully_connected_target_509:
��)
fully_connected_target_511:	�
	dense_515:
��
	dense_517:	�
dense_1_520:
��
dense_1_522:	�
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�.embedding_layer_origin/StatefulPartitionedCall�.embedding_layer_target/StatefulPartitionedCall�.fully_connected_source/StatefulPartitionedCall�.fully_connected_target/StatefulPartitionedCall�
.embedding_layer_target/StatefulPartitionedCallStatefulPartitionedCallinputs_1embedding_layer_target_496*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_embedding_layer_target_layer_call_and_return_conditional_losses_24120
.embedding_layer_target/StatefulPartitionedCall�
.embedding_layer_origin/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_layer_origin_499*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_embedding_layer_origin_layer_call_and_return_conditional_losses_25520
.embedding_layer_origin/StatefulPartitionedCall�
flatten_1/PartitionedCallPartitionedCall7embedding_layer_target/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_flatten_1_layer_call_and_return_conditional_losses_2652
flatten_1/PartitionedCall�
flatten/PartitionedCallPartitionedCall7embedding_layer_origin/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_2732
flatten/PartitionedCall�
.fully_connected_source/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0fully_connected_source_504fully_connected_source_506*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_fully_connected_source_layer_call_and_return_conditional_losses_28620
.fully_connected_source/StatefulPartitionedCall�
.fully_connected_target/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0fully_connected_target_509fully_connected_target_511*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_fully_connected_target_layer_call_and_return_conditional_losses_30320
.fully_connected_target/StatefulPartitionedCall�
concatenate/PartitionedCallPartitionedCall7fully_connected_source/StatefulPartitionedCall:output:07fully_connected_target/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_concatenate_layer_call_and_return_conditional_losses_3162
concatenate/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0	dense_515	dense_517*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_3292
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_520dense_1_522*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_3452!
dense_1/StatefulPartitionedCall�
softmax/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_softmax_layer_call_and_return_conditional_losses_3562
softmax/PartitionedCall�
IdentityIdentity softmax/PartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall/^embedding_layer_origin/StatefulPartitionedCall/^embedding_layer_target/StatefulPartitionedCall/^fully_connected_source/StatefulPartitionedCall/^fully_connected_target/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::���������:���������: : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2`
.embedding_layer_origin/StatefulPartitionedCall.embedding_layer_origin/StatefulPartitionedCall2`
.embedding_layer_target/StatefulPartitionedCall.embedding_layer_target/StatefulPartitionedCall2`
.fully_connected_source/StatefulPartitionedCall.fully_connected_source/StatefulPartitionedCall2`
.fully_connected_target/StatefulPartitionedCall.fully_connected_target/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
O__inference_embedding_layer_target_layer_call_and_return_conditional_losses_241

inputs(
embedding_lookup_235:
��
identity��embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:���������2
Cast�
embedding_lookupResourceGatherembedding_lookup_235Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*'
_class
loc:@embedding_lookup/235*,
_output_shapes
:����������*
dtype02
embedding_lookup�
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@embedding_lookup/235*,
_output_shapes
:����������2
embedding_lookup/Identity�
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:����������2
embedding_lookup/Identity_1�
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
@__inference_dense_1_layer_call_and_return_conditional_losses_345

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
p
D__inference_concatenate_layer_call_and_return_conditional_losses_937
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:����������2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������:����������:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/1
�Q
�	
__inference__wrapped_model_222
input_source
input_targetE
1model_embedding_layer_target_embedding_lookup_176:
��E
1model_embedding_layer_origin_embedding_lookup_182:
�(�O
;model_fully_connected_source_matmul_readvariableop_resource:
��K
<model_fully_connected_source_biasadd_readvariableop_resource:	�O
;model_fully_connected_target_matmul_readvariableop_resource:
��K
<model_fully_connected_target_biasadd_readvariableop_resource:	�>
*model_dense_matmul_readvariableop_resource:
��:
+model_dense_biasadd_readvariableop_resource:	�@
,model_dense_1_matmul_readvariableop_resource:
��<
-model_dense_1_biasadd_readvariableop_resource:	�
identity��"model/dense/BiasAdd/ReadVariableOp�!model/dense/MatMul/ReadVariableOp�$model/dense_1/BiasAdd/ReadVariableOp�#model/dense_1/MatMul/ReadVariableOp�-model/embedding_layer_origin/embedding_lookup�-model/embedding_layer_target/embedding_lookup�3model/fully_connected_source/BiasAdd/ReadVariableOp�2model/fully_connected_source/MatMul/ReadVariableOp�3model/fully_connected_target/BiasAdd/ReadVariableOp�2model/fully_connected_target/MatMul/ReadVariableOp�
!model/embedding_layer_target/CastCastinput_target*

DstT0*

SrcT0*'
_output_shapes
:���������2#
!model/embedding_layer_target/Cast�
-model/embedding_layer_target/embedding_lookupResourceGather1model_embedding_layer_target_embedding_lookup_176%model/embedding_layer_target/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*D
_class:
86loc:@model/embedding_layer_target/embedding_lookup/176*,
_output_shapes
:����������*
dtype02/
-model/embedding_layer_target/embedding_lookup�
6model/embedding_layer_target/embedding_lookup/IdentityIdentity6model/embedding_layer_target/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@model/embedding_layer_target/embedding_lookup/176*,
_output_shapes
:����������28
6model/embedding_layer_target/embedding_lookup/Identity�
8model/embedding_layer_target/embedding_lookup/Identity_1Identity?model/embedding_layer_target/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:����������2:
8model/embedding_layer_target/embedding_lookup/Identity_1�
!model/embedding_layer_origin/CastCastinput_source*

DstT0*

SrcT0*'
_output_shapes
:���������2#
!model/embedding_layer_origin/Cast�
-model/embedding_layer_origin/embedding_lookupResourceGather1model_embedding_layer_origin_embedding_lookup_182%model/embedding_layer_origin/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*D
_class:
86loc:@model/embedding_layer_origin/embedding_lookup/182*,
_output_shapes
:����������*
dtype02/
-model/embedding_layer_origin/embedding_lookup�
6model/embedding_layer_origin/embedding_lookup/IdentityIdentity6model/embedding_layer_origin/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@model/embedding_layer_origin/embedding_lookup/182*,
_output_shapes
:����������28
6model/embedding_layer_origin/embedding_lookup/Identity�
8model/embedding_layer_origin/embedding_lookup/Identity_1Identity?model/embedding_layer_origin/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:����������2:
8model/embedding_layer_origin/embedding_lookup/Identity_1
model/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
model/flatten_1/Const�
model/flatten_1/ReshapeReshapeAmodel/embedding_layer_target/embedding_lookup/Identity_1:output:0model/flatten_1/Const:output:0*
T0*(
_output_shapes
:����������2
model/flatten_1/Reshape{
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����X  2
model/flatten/Const�
model/flatten/ReshapeReshapeAmodel/embedding_layer_origin/embedding_lookup/Identity_1:output:0model/flatten/Const:output:0*
T0*(
_output_shapes
:����������2
model/flatten/Reshape�
2model/fully_connected_source/MatMul/ReadVariableOpReadVariableOp;model_fully_connected_source_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype024
2model/fully_connected_source/MatMul/ReadVariableOp�
#model/fully_connected_source/MatMulMatMulmodel/flatten/Reshape:output:0:model/fully_connected_source/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#model/fully_connected_source/MatMul�
3model/fully_connected_source/BiasAdd/ReadVariableOpReadVariableOp<model_fully_connected_source_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype025
3model/fully_connected_source/BiasAdd/ReadVariableOp�
$model/fully_connected_source/BiasAddBiasAdd-model/fully_connected_source/MatMul:product:0;model/fully_connected_source/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2&
$model/fully_connected_source/BiasAdd�
!model/fully_connected_source/ReluRelu-model/fully_connected_source/BiasAdd:output:0*
T0*(
_output_shapes
:����������2#
!model/fully_connected_source/Relu�
2model/fully_connected_target/MatMul/ReadVariableOpReadVariableOp;model_fully_connected_target_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype024
2model/fully_connected_target/MatMul/ReadVariableOp�
#model/fully_connected_target/MatMulMatMul model/flatten_1/Reshape:output:0:model/fully_connected_target/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#model/fully_connected_target/MatMul�
3model/fully_connected_target/BiasAdd/ReadVariableOpReadVariableOp<model_fully_connected_target_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype025
3model/fully_connected_target/BiasAdd/ReadVariableOp�
$model/fully_connected_target/BiasAddBiasAdd-model/fully_connected_target/MatMul:product:0;model/fully_connected_target/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2&
$model/fully_connected_target/BiasAdd�
!model/fully_connected_target/ReluRelu-model/fully_connected_target/BiasAdd:output:0*
T0*(
_output_shapes
:����������2#
!model/fully_connected_target/Relu�
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
model/concatenate/concat/axis�
model/concatenate/concatConcatV2/model/fully_connected_source/Relu:activations:0/model/fully_connected_target/Relu:activations:0&model/concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������2
model/concatenate/concat�
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02#
!model/dense/MatMul/ReadVariableOp�
model/dense/MatMulMatMul!model/concatenate/concat:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
model/dense/MatMul�
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02$
"model/dense/BiasAdd/ReadVariableOp�
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
model/dense/BiasAdd}
model/dense/ReluRelumodel/dense/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
model/dense/Relu�
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02%
#model/dense_1/MatMul/ReadVariableOp�
model/dense_1/MatMulMatMulmodel/dense/Relu:activations:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
model/dense_1/MatMul�
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02&
$model/dense_1/BiasAdd/ReadVariableOp�
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
model/dense_1/BiasAdd�
model/softmax/SoftmaxSoftmaxmodel/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
model/softmax/Softmax�
IdentityIdentitymodel/softmax/Softmax:softmax:0#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp.^model/embedding_layer_origin/embedding_lookup.^model/embedding_layer_target/embedding_lookup4^model/fully_connected_source/BiasAdd/ReadVariableOp3^model/fully_connected_source/MatMul/ReadVariableOp4^model/fully_connected_target/BiasAdd/ReadVariableOp3^model/fully_connected_target/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::���������:���������: : : : : : : : : : 2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2^
-model/embedding_layer_origin/embedding_lookup-model/embedding_layer_origin/embedding_lookup2^
-model/embedding_layer_target/embedding_lookup-model/embedding_layer_target/embedding_lookup2j
3model/fully_connected_source/BiasAdd/ReadVariableOp3model/fully_connected_source/BiasAdd/ReadVariableOp2h
2model/fully_connected_source/MatMul/ReadVariableOp2model/fully_connected_source/MatMul/ReadVariableOp2j
3model/fully_connected_target/BiasAdd/ReadVariableOp3model/fully_connected_target/BiasAdd/ReadVariableOp2h
2model/fully_connected_target/MatMul/ReadVariableOp2model/fully_connected_target/MatMul/ReadVariableOp:U Q
'
_output_shapes
:���������
&
_user_specified_nameinput_source:UQ
'
_output_shapes
:���������
&
_user_specified_nameinput_target"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
E
input_source5
serving_default_input_source:0���������
E
input_target5
serving_default_input_target:0���������<
softmax1
StatefulPartitionedCall:0����������tensorflow/serving/predict:��
�[
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer-11
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
�_default_save_signature
�__call__
+�&call_and_return_all_conditional_losses"�W
_tf_keras_network�W{"name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_source"}, "name": "input_source", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_target"}, "name": "input_target", "inbound_nodes": []}, {"class_name": "Embedding", "config": {"name": "embedding_layer_origin", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "input_dim": 5190, "output_dim": 200, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 3}, "name": "embedding_layer_origin", "inbound_nodes": [[["input_source", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_layer_target", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 3933, "output_dim": 200, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "embedding_layer_target", "inbound_nodes": [[["input_target", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["embedding_layer_origin", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["embedding_layer_target", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fully_connected_source", "trainable": true, "dtype": "float32", "units": 200, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fully_connected_source", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fully_connected_target", "trainable": true, "dtype": "float32", "units": 200, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fully_connected_target", "inbound_nodes": [[["flatten_1", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["fully_connected_source", 0, 0, {}], ["fully_connected_target", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 200, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 3933, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Softmax", "config": {"name": "softmax", "trainable": true, "dtype": "float32", "axis": -1}, "name": "softmax", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}], "input_layers": [["input_source", 0, 0], ["input_target", 0, 0]], "output_layers": [["softmax", 0, 0]]}, "shared_object_id": 22, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 3]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 3]}, {"class_name": "TensorShape", "items": [null, 1]}], "is_graph_network": true, "save_spec": [{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 3]}, "float32", "input_source"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "input_target"]}], "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_source"}, "name": "input_source", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_target"}, "name": "input_target", "inbound_nodes": [], "shared_object_id": 1}, {"class_name": "Embedding", "config": {"name": "embedding_layer_origin", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "input_dim": 5190, "output_dim": 200, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 2}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 3}, "name": "embedding_layer_origin", "inbound_nodes": [[["input_source", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "Embedding", "config": {"name": "embedding_layer_target", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 3933, "output_dim": 200, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 4}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "embedding_layer_target", "inbound_nodes": [[["input_target", 0, 0, {}]]], "shared_object_id": 5}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["embedding_layer_origin", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["embedding_layer_target", 0, 0, {}]]], "shared_object_id": 7}, {"class_name": "Dense", "config": {"name": "fully_connected_source", "trainable": true, "dtype": "float32", "units": 200, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fully_connected_source", "inbound_nodes": [[["flatten", 0, 0, {}]]], "shared_object_id": 10}, {"class_name": "Dense", "config": {"name": "fully_connected_target", "trainable": true, "dtype": "float32", "units": 200, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fully_connected_target", "inbound_nodes": [[["flatten_1", 0, 0, {}]]], "shared_object_id": 13}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["fully_connected_source", 0, 0, {}], ["fully_connected_target", 0, 0, {}]]], "shared_object_id": 14}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 200, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["concatenate", 0, 0, {}]]], "shared_object_id": 17}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 3933, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]], "shared_object_id": 20}, {"class_name": "Softmax", "config": {"name": "softmax", "trainable": true, "dtype": "float32", "axis": -1}, "name": "softmax", "inbound_nodes": [[["dense_1", 0, 0, {}]]], "shared_object_id": 21}], "input_layers": [["input_source", 0, 0], ["input_target", 0, 0]], "output_layers": [["softmax", 0, 0]]}}, "training_config": {"loss": "sparse_categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "sparse_categorical_accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}, "shared_object_id": 25}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.001, "decay": 0.0, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_source", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_source"}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_target", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_target"}}
�

embeddings
regularization_losses
trainable_variables
	variables
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "embedding_layer_origin", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "stateful": false, "must_restore_from_config": false, "class_name": "Embedding", "config": {"name": "embedding_layer_origin", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "input_dim": 5190, "output_dim": 200, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 2}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 3}, "inbound_nodes": [[["input_source", 0, 0, {}]]], "shared_object_id": 3, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}}
�

embeddings
regularization_losses
trainable_variables
	variables
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "embedding_layer_target", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "stateful": false, "must_restore_from_config": false, "class_name": "Embedding", "config": {"name": "embedding_layer_target", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 3933, "output_dim": 200, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 4}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "inbound_nodes": [[["input_target", 0, 0, {}]]], "shared_object_id": 5, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
�
regularization_losses
trainable_variables
	variables
 	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["embedding_layer_origin", 0, 0, {}]]], "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 26}}
�
!regularization_losses
"trainable_variables
#	variables
$	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["embedding_layer_target", 0, 0, {}]]], "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 27}}
�	

%kernel
&bias
'regularization_losses
(trainable_variables
)	variables
*	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "fully_connected_source", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "fully_connected_source", "trainable": true, "dtype": "float32", "units": 200, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["flatten", 0, 0, {}]]], "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 600}}, "shared_object_id": 28}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 600]}}
�	

+kernel
,bias
-regularization_losses
.trainable_variables
/	variables
0	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "fully_connected_target", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "fully_connected_target", "trainable": true, "dtype": "float32", "units": 200, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["flatten_1", 0, 0, {}]]], "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 200}}, "shared_object_id": 29}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200]}}
�
1regularization_losses
2trainable_variables
3	variables
4	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "concatenate", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "inbound_nodes": [[["fully_connected_source", 0, 0, {}], ["fully_connected_target", 0, 0, {}]]], "shared_object_id": 14, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 200]}, {"class_name": "TensorShape", "items": [null, 200]}]}
�

5kernel
6bias
7regularization_losses
8trainable_variables
9	variables
:	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 200, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["concatenate", 0, 0, {}]]], "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 400}}, "shared_object_id": 30}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 400]}}
�

;kernel
<bias
=regularization_losses
>trainable_variables
?	variables
@	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 3933, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dense", 0, 0, {}]]], "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 200}}, "shared_object_id": 31}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200]}}
�
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "softmax", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Softmax", "config": {"name": "softmax", "trainable": true, "dtype": "float32", "axis": -1}, "inbound_nodes": [[["dense_1", 0, 0, {}]]], "shared_object_id": 21}
"
	optimizer
 "
trackable_list_wrapper
f
0
1
%2
&3
+4
,5
56
67
;8
<9"
trackable_list_wrapper
f
0
1
%2
&3
+4
,5
56
67
;8
<9"
trackable_list_wrapper
�
Elayer_regularization_losses
Fmetrics
Glayer_metrics
regularization_losses
Hnon_trainable_variables

Ilayers
trainable_variables
	variables
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
5:3
�(�2!embedding_layer_origin/embeddings
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
�
Jlayer_regularization_losses
Kmetrics
Llayer_metrics
regularization_losses
Mnon_trainable_variables

Nlayers
trainable_variables
	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
5:3
��2!embedding_layer_target/embeddings
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
�
Olayer_regularization_losses
Pmetrics
Qlayer_metrics
regularization_losses
Rnon_trainable_variables

Slayers
trainable_variables
	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Tlayer_regularization_losses
Umetrics
Vlayer_metrics
regularization_losses
Wnon_trainable_variables

Xlayers
trainable_variables
	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Ylayer_regularization_losses
Zmetrics
[layer_metrics
!regularization_losses
\non_trainable_variables

]layers
"trainable_variables
#	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
1:/
��2fully_connected_source/kernel
*:(�2fully_connected_source/bias
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
�
^layer_regularization_losses
_metrics
`layer_metrics
'regularization_losses
anon_trainable_variables

blayers
(trainable_variables
)	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
1:/
��2fully_connected_target/kernel
*:(�2fully_connected_target/bias
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
�
clayer_regularization_losses
dmetrics
elayer_metrics
-regularization_losses
fnon_trainable_variables

glayers
.trainable_variables
/	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
hlayer_regularization_losses
imetrics
jlayer_metrics
1regularization_losses
knon_trainable_variables

llayers
2trainable_variables
3	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 :
��2dense/kernel
:�2
dense/bias
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
�
mlayer_regularization_losses
nmetrics
olayer_metrics
7regularization_losses
pnon_trainable_variables

qlayers
8trainable_variables
9	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
": 
��2dense_1/kernel
:�2dense_1/bias
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
�
rlayer_regularization_losses
smetrics
tlayer_metrics
=regularization_losses
unon_trainable_variables

vlayers
>trainable_variables
?	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
wlayer_regularization_losses
xmetrics
ylayer_metrics
Aregularization_losses
znon_trainable_variables

{layers
Btrainable_variables
C	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
v
0
1
2
3
4
5
6
7
	8

9
10
11"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
	~total
	count
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 32}
�

�total

�count
�
_fn_kwargs
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "name": "sparse_categorical_accuracy", "dtype": "float32", "config": {"name": "sparse_categorical_accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}, "shared_object_id": 25}
:  (2total
:  (2count
.
~0
1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
�2�
__inference__wrapped_model_222�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *X�U
S�P
&�#
input_source���������
&�#
input_target���������
�2�
#__inference_model_layer_call_fn_382
#__inference_model_layer_call_fn_700
#__inference_model_layer_call_fn_726
#__inference_model_layer_call_fn_576�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
>__inference_model_layer_call_and_return_conditional_losses_777
>__inference_model_layer_call_and_return_conditional_losses_828
>__inference_model_layer_call_and_return_conditional_losses_611
>__inference_model_layer_call_and_return_conditional_losses_646�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
4__inference_embedding_layer_origin_layer_call_fn_835�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
O__inference_embedding_layer_origin_layer_call_and_return_conditional_losses_845�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
4__inference_embedding_layer_target_layer_call_fn_852�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
O__inference_embedding_layer_target_layer_call_and_return_conditional_losses_862�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
%__inference_flatten_layer_call_fn_867�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
@__inference_flatten_layer_call_and_return_conditional_losses_873�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_flatten_1_layer_call_fn_878�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_flatten_1_layer_call_and_return_conditional_losses_884�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
4__inference_fully_connected_source_layer_call_fn_893�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
O__inference_fully_connected_source_layer_call_and_return_conditional_losses_904�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
4__inference_fully_connected_target_layer_call_fn_913�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
O__inference_fully_connected_target_layer_call_and_return_conditional_losses_924�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_concatenate_layer_call_fn_930�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_concatenate_layer_call_and_return_conditional_losses_937�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
#__inference_dense_layer_call_fn_946�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
>__inference_dense_layer_call_and_return_conditional_losses_957�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
%__inference_dense_1_layer_call_fn_966�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
@__inference_dense_1_layer_call_and_return_conditional_losses_976�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
%__inference_softmax_layer_call_fn_981�
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
@__inference_softmax_layer_call_and_return_conditional_losses_986�
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
!__inference_signature_wrapper_674input_sourceinput_target"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
__inference__wrapped_model_222�
%&+,56;<b�_
X�U
S�P
&�#
input_source���������
&�#
input_target���������
� "2�/
-
softmax"�
softmax�����������
D__inference_concatenate_layer_call_and_return_conditional_losses_937�\�Y
R�O
M�J
#� 
inputs/0����������
#� 
inputs/1����������
� "&�#
�
0����������
� �
)__inference_concatenate_layer_call_fn_930y\�Y
R�O
M�J
#� 
inputs/0����������
#� 
inputs/1����������
� "������������
@__inference_dense_1_layer_call_and_return_conditional_losses_976^;<0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� z
%__inference_dense_1_layer_call_fn_966Q;<0�-
&�#
!�
inputs����������
� "������������
>__inference_dense_layer_call_and_return_conditional_losses_957^560�-
&�#
!�
inputs����������
� "&�#
�
0����������
� x
#__inference_dense_layer_call_fn_946Q560�-
&�#
!�
inputs����������
� "������������
O__inference_embedding_layer_origin_layer_call_and_return_conditional_losses_845`/�,
%�"
 �
inputs���������
� "*�'
 �
0����������
� �
4__inference_embedding_layer_origin_layer_call_fn_835S/�,
%�"
 �
inputs���������
� "������������
O__inference_embedding_layer_target_layer_call_and_return_conditional_losses_862`/�,
%�"
 �
inputs���������
� "*�'
 �
0����������
� �
4__inference_embedding_layer_target_layer_call_fn_852S/�,
%�"
 �
inputs���������
� "������������
B__inference_flatten_1_layer_call_and_return_conditional_losses_884^4�1
*�'
%�"
inputs����������
� "&�#
�
0����������
� |
'__inference_flatten_1_layer_call_fn_878Q4�1
*�'
%�"
inputs����������
� "������������
@__inference_flatten_layer_call_and_return_conditional_losses_873^4�1
*�'
%�"
inputs����������
� "&�#
�
0����������
� z
%__inference_flatten_layer_call_fn_867Q4�1
*�'
%�"
inputs����������
� "������������
O__inference_fully_connected_source_layer_call_and_return_conditional_losses_904^%&0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
4__inference_fully_connected_source_layer_call_fn_893Q%&0�-
&�#
!�
inputs����������
� "������������
O__inference_fully_connected_target_layer_call_and_return_conditional_losses_924^+,0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
4__inference_fully_connected_target_layer_call_fn_913Q+,0�-
&�#
!�
inputs����������
� "������������
>__inference_model_layer_call_and_return_conditional_losses_611�
%&+,56;<j�g
`�]
S�P
&�#
input_source���������
&�#
input_target���������
p 

 
� "&�#
�
0����������
� �
>__inference_model_layer_call_and_return_conditional_losses_646�
%&+,56;<j�g
`�]
S�P
&�#
input_source���������
&�#
input_target���������
p

 
� "&�#
�
0����������
� �
>__inference_model_layer_call_and_return_conditional_losses_777�
%&+,56;<b�_
X�U
K�H
"�
inputs/0���������
"�
inputs/1���������
p 

 
� "&�#
�
0����������
� �
>__inference_model_layer_call_and_return_conditional_losses_828�
%&+,56;<b�_
X�U
K�H
"�
inputs/0���������
"�
inputs/1���������
p

 
� "&�#
�
0����������
� �
#__inference_model_layer_call_fn_382�
%&+,56;<j�g
`�]
S�P
&�#
input_source���������
&�#
input_target���������
p 

 
� "������������
#__inference_model_layer_call_fn_576�
%&+,56;<j�g
`�]
S�P
&�#
input_source���������
&�#
input_target���������
p

 
� "������������
#__inference_model_layer_call_fn_700�
%&+,56;<b�_
X�U
K�H
"�
inputs/0���������
"�
inputs/1���������
p 

 
� "������������
#__inference_model_layer_call_fn_726�
%&+,56;<b�_
X�U
K�H
"�
inputs/0���������
"�
inputs/1���������
p

 
� "������������
!__inference_signature_wrapper_674�
%&+,56;<}�z
� 
s�p
6
input_source&�#
input_source���������
6
input_target&�#
input_target���������"2�/
-
softmax"�
softmax�����������
@__inference_softmax_layer_call_and_return_conditional_losses_986^4�1
*�'
!�
inputs����������

 
� "&�#
�
0����������
� z
%__inference_softmax_layer_call_fn_981Q4�1
*�'
!�
inputs����������

 
� "�����������