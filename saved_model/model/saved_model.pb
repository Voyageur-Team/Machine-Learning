��	
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
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
$
DisableCopyOnRead
resource�
�
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype�
.
Identity

input"T
output"T"	
Ttype
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype�
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype�
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
�
TopKV2

input"T
k"Tk
values"T
indices"
index_type"
sortedbool("
Ttype:
2	"
Tktype0:
2	"

index_typetype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.15.02v2.15.0-rc1-8-g6887368d6d48��
~
ConstConst*
_output_shapes
:*
dtype0	*E
value<B:	"0                                          
{
Const_1Const*
_output_shapes
:*
dtype0*@
value7B5B0-25B100-200B200+B25-50B50-100BUnknown
�
Const_2Const*
_output_shapes
:*
dtype0	*�
value�B�	"�                                                        	       
                                                                                                                                     
�
Const_3Const*
_output_shapes
:*
dtype0*�
value�B�BAlamBBahariBBelanjaBBudayaBEdukasiBGunungBHealingBKebun BinatangBKlubBKulinerBLainnyaBPantaiBPusat PerbelanjaanBReligiBSejarahB	Spot FotoBTamanBTaman HiburanB
Taman KotaBWahanaB
Wahana AirBWahana AlamBWahana BermainBWahana EkstremBWisata BermainBWisata RekreasiBnanBpantai
�
Const_4Const*
_output_shapes
:*
dtype0	*}
valuetBr	"h                                                        	       
                            
�
Const_5Const*
_output_shapes
:*
dtype0*�
valuexBvBBaliBBandungBBogorBJakartaBLombokBMalangBManadoB
Raja AmpatBSemarangBSoloBSurabayaB
YogyakartaBnan
I
Const_6Const*
_output_shapes
: *
dtype0	*
value	B	 R 
I
Const_7Const*
_output_shapes
: *
dtype0	*
value	B	 R 
I
Const_8Const*
_output_shapes
: *
dtype0	*
value	B	 R 
n

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name101431*
value_dtype0	
p
hash_table_1HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name101408*
value_dtype0	
p
hash_table_2HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name101385*
value_dtype0	
�
dense_2/biasVarHandleOp*
_output_shapes
: *

debug_namedense_2/bias/*
dtype0*
shape: *
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
: *
dtype0
�
dense_2/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_2/kernel/*
dtype0*
shape
:` *
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:` *
dtype0
�
embedding_7/embeddingsVarHandleOp*
_output_shapes
: *'

debug_nameembedding_7/embeddings/*
dtype0*
shape
: *'
shared_nameembedding_7/embeddings
�
*embedding_7/embeddings/Read/ReadVariableOpReadVariableOpembedding_7/embeddings*
_output_shapes

: *
dtype0
�
embedding_6/embeddingsVarHandleOp*
_output_shapes
: *'

debug_nameembedding_6/embeddings/*
dtype0*
shape
: *'
shared_nameembedding_6/embeddings
�
*embedding_6/embeddings/Read/ReadVariableOpReadVariableOpembedding_6/embeddings*
_output_shapes

: *
dtype0
�
embedding_5/embeddingsVarHandleOp*
_output_shapes
: *'

debug_nameembedding_5/embeddings/*
dtype0*
shape
: *'
shared_nameembedding_5/embeddings
�
*embedding_5/embeddings/Read/ReadVariableOpReadVariableOpembedding_5/embeddings*
_output_shapes

: *
dtype0
�

candidatesVarHandleOp*
_output_shapes
: *

debug_namecandidates/*
dtype0*
shape:	�	 *
shared_name
candidates
j
candidates/Read/ReadVariableOpReadVariableOp
candidates*
_output_shapes
:	�	 *
dtype0
�
identifiersVarHandleOp*
_output_shapes
: *

debug_nameidentifiers/*
dtype0*
shape:�	*
shared_nameidentifiers
h
identifiers/Read/ReadVariableOpReadVariableOpidentifiers*
_output_shapes	
:�	*
dtype0
s
serving_default_CategoryPlaceholder*#
_output_shapes
:���������*
dtype0*
shape:���������
o
serving_default_CityPlaceholder*#
_output_shapes
:���������*
dtype0*
shape:���������
y
serving_default_Price_CategoryPlaceholder*#
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_Categoryserving_default_Cityserving_default_Price_Categoryhash_table_2Const_8embedding_5/embeddingshash_table_1Const_7embedding_6/embeddings
hash_tableConst_6embedding_7/embeddingsdense_2/kerneldense_2/bias
candidatesidentifiers*
Tin
2			*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������
:���������
*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_481334
�
StatefulPartitionedCall_1StatefulPartitionedCallhash_table_2Const_5Const_4*
Tin
2	*
Tout
2*
_collective_manager_ids
 *&
 _has_manual_control_dependencies(*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__initializer_481409
�
StatefulPartitionedCall_2StatefulPartitionedCallhash_table_1Const_3Const_2*
Tin
2	*
Tout
2*
_collective_manager_ids
 *&
 _has_manual_control_dependencies(*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__initializer_481424
�
StatefulPartitionedCall_3StatefulPartitionedCall
hash_tableConst_1Const*
Tin
2	*
Tout
2*
_collective_manager_ids
 *&
 _has_manual_control_dependencies(*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__initializer_481439
`
NoOpNoOp^StatefulPartitionedCall_1^StatefulPartitionedCall_2^StatefulPartitionedCall_3
�G
Const_9Const"/device:CPU:0*
_output_shapes
: *
dtype0*�F
value�FB�F B�F
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
query_model
	identifiers
	_identifiers


candidates

_candidates
query_with_exclusions

signatures*
5
0
1
2
3
4
	5

6*
'
0
1
2
3
4*
* 
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
/
	capture_1
	capture_4
	capture_7* 
�
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses
$user_embedding_model
%dense_layers*
KE
VARIABLE_VALUEidentifiers&identifiers/.ATTRIBUTES/VARIABLE_VALUE*
IC
VARIABLE_VALUE
candidates%candidates/.ATTRIBUTES/VARIABLE_VALUE*
* 

&serving_default* 
VP
VARIABLE_VALUEembedding_5/embeddings&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEembedding_6/embeddings&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEembedding_7/embeddings&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_2/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEdense_2/bias&variables/4/.ATTRIBUTES/VARIABLE_VALUE*

	0

1*

0*
* 
* 
* 
/
	capture_1
	capture_4
	capture_7* 
/
	capture_1
	capture_4
	capture_7* 
/
	capture_1
	capture_4
	capture_7* 
/
	capture_1
	capture_4
	capture_7* 
* 
* 
* 
'
0
1
2
3
4*
'
0
1
2
3
4*
* 
�
'non_trainable_variables

(layers
)metrics
*layer_regularization_losses
+layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*

,trace_0
-trace_1* 

.trace_0
/trace_1* 
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses
6city_embedding
7category_embedding
8price_embedding*
�
9layer_with_weights-0
9layer-0
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses*
/
	capture_1
	capture_4
	capture_7* 
* 

$0
%1*
* 
* 
* 
/
	capture_1
	capture_4
	capture_7* 
/
	capture_1
	capture_4
	capture_7* 
/
	capture_1
	capture_4
	capture_7* 
/
	capture_1
	capture_4
	capture_7* 

0
1
2*

0
1
2*
* 
�
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*

Etrace_0
Ftrace_1* 

Gtrace_0
Htrace_1* 
�
Ilayer-0
Jlayer_with_weights-0
Jlayer-1
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses*
�
Qlayer-0
Rlayer_with_weights-0
Rlayer-1
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses*
�
Ylayer-0
Zlayer_with_weights-0
Zlayer-1
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses*
�
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses

kernel
bias*

0
1*

0
1*
* 
�
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses*

ltrace_0
mtrace_1* 

ntrace_0
otrace_1* 
* 

60
71
82*
* 
* 
* 
/
	capture_1
	capture_4
	capture_7* 
/
	capture_1
	capture_4
	capture_7* 
/
	capture_1
	capture_4
	capture_7* 
/
	capture_1
	capture_4
	capture_7* 
#
p	keras_api
qlookup_table* 
�
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses

embeddings*

0*

0*
* 
�
xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses*

}trace_0
~trace_1* 

trace_0
�trace_1* 
%
�	keras_api
�lookup_table* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

embeddings*

0*

0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
%
�	keras_api
�lookup_table* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

embeddings*

0*

0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

90*
* 
* 
* 
* 
* 
* 
* 
* 
V
�_initializer
�_create_resource
�_initialize
�_destroy_resource* 

0*

0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

I0
J1*
* 
* 
* 

	capture_1* 

	capture_1* 

	capture_1* 

	capture_1* 
* 
V
�_initializer
�_create_resource
�_initialize
�_destroy_resource* 

0*

0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

Q0
R1*
* 
* 
* 

	capture_1* 

	capture_1* 

	capture_1* 

	capture_1* 
* 
V
�_initializer
�_create_resource
�_initialize
�_destroy_resource* 

0*

0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

Y0
Z1*
* 
* 
* 

	capture_1* 

	capture_1* 

	capture_1* 

	capture_1* 
* 
* 
* 
* 
* 
* 
* 
* 

�trace_0* 

�trace_0* 

�trace_0* 
* 
* 
* 
* 
* 
* 
* 
* 

�trace_0* 

�trace_0* 

�trace_0* 
* 
* 
* 
* 
* 
* 
* 
* 

�trace_0* 

�trace_0* 

�trace_0* 
* 
* 
* 
* 
* 
* 
* 
* 
"
�	capture_1
�	capture_2* 
* 
* 
"
�	capture_1
�	capture_2* 
* 
* 
"
�	capture_1
�	capture_2* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_4StatefulPartitionedCallsaver_filenameidentifiers
candidatesembedding_5/embeddingsembedding_6/embeddingsembedding_7/embeddingsdense_2/kerneldense_2/biasConst_9*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__traced_save_481525
�
StatefulPartitionedCall_5StatefulPartitionedCallsaver_filenameidentifiers
candidatesembedding_5/embeddingsembedding_6/embeddingsembedding_7/embeddingsdense_2/kerneldense_2/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__traced_restore_481555��
�B
�
__inference__traced_save_481525
file_prefix1
"read_disablecopyonread_identifiers:	�	6
#read_1_disablecopyonread_candidates:	�	 A
/read_2_disablecopyonread_embedding_5_embeddings: A
/read_3_disablecopyonread_embedding_6_embeddings: A
/read_4_disablecopyonread_embedding_7_embeddings: 9
'read_5_disablecopyonread_dense_2_kernel:` 3
%read_6_disablecopyonread_dense_2_bias: 
savev2_const_9
identity_15��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: t
Read/DisableCopyOnReadDisableCopyOnRead"read_disablecopyonread_identifiers"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp"read_disablecopyonread_identifiers^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�	*
dtype0f
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�	^

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes	
:�	w
Read_1/DisableCopyOnReadDisableCopyOnRead#read_1_disablecopyonread_candidates"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp#read_1_disablecopyonread_candidates^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�	 *
dtype0n

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�	 d

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:	�	 �
Read_2/DisableCopyOnReadDisableCopyOnRead/read_2_disablecopyonread_embedding_5_embeddings"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp/read_2_disablecopyonread_embedding_5_embeddings^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0m

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: c

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_3/DisableCopyOnReadDisableCopyOnRead/read_3_disablecopyonread_embedding_6_embeddings"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp/read_3_disablecopyonread_embedding_6_embeddings^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0m

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: c

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_4/DisableCopyOnReadDisableCopyOnRead/read_4_disablecopyonread_embedding_7_embeddings"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp/read_4_disablecopyonread_embedding_7_embeddings^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0m

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: c

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes

: {
Read_5/DisableCopyOnReadDisableCopyOnRead'read_5_disablecopyonread_dense_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp'read_5_disablecopyonread_dense_2_kernel^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:` *
dtype0n
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:` e
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes

:` y
Read_6/DisableCopyOnReadDisableCopyOnRead%read_6_disablecopyonread_dense_2_bias"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp%read_6_disablecopyonread_dense_2_bias^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0j
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&identifiers/.ATTRIBUTES/VARIABLE_VALUEB%candidates/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH}
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0savev2_const_9"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes

2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_14Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_15IdentityIdentity_14:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp*
_output_shapes
 "#
identity_15Identity_15:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp:?;

_output_shapes
: 
!
_user_specified_name	Const_9:,(
&
_user_specified_namedense_2/bias:.*
(
_user_specified_namedense_2/kernel:62
0
_user_specified_nameembedding_7/embeddings:62
0
_user_specified_nameembedding_6/embeddings:62
0
_user_specified_nameembedding_5/embeddings:*&
$
_user_specified_name
candidates:+'
%
_user_specified_nameidentifiers:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
.__inference_query_model_1_layer_call_fn_481152
category
city
price_category
unknown
	unknown_0	
	unknown_1: 
	unknown_2
	unknown_3	
	unknown_4: 
	unknown_5
	unknown_6	
	unknown_7: 
	unknown_8:` 
	unknown_9: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallcategorycityprice_categoryunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2			*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_query_model_1_layer_call_and_return_conditional_losses_481094o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������:���������:���������: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name481148:&"
 
_user_specified_name481146:&"
 
_user_specified_name481144:


_output_shapes
: :&	"
 
_user_specified_name481140:&"
 
_user_specified_name481138:

_output_shapes
: :&"
 
_user_specified_name481134:&"
 
_user_specified_name481132:

_output_shapes
: :&"
 
_user_specified_name481128:SO
#
_output_shapes
:���������
(
_user_specified_namePrice_Category:IE
#
_output_shapes
:���������

_user_specified_nameCity:M I
#
_output_shapes
:���������
"
_user_specified_name
Category
�	
�
C__inference_dense_2_layer_call_and_return_conditional_losses_481353

inputs0
matmul_readvariableop_resource:` -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:` *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:��������� S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�
�
-__inference_sequential_8_layer_call_fn_480794
string_lookup_6_input
unknown
	unknown_0	
	unknown_1: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallstring_lookup_6_inputunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_8_layer_call_and_return_conditional_losses_480772o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name480790:

_output_shapes
: :&"
 
_user_specified_name480786:Z V
#
_output_shapes
:���������
/
_user_specified_namestring_lookup_6_input
�
�
.__inference_query_model_1_layer_call_fn_481123
category
city
price_category
unknown
	unknown_0	
	unknown_1: 
	unknown_2
	unknown_3	
	unknown_4: 
	unknown_5
	unknown_6	
	unknown_7: 
	unknown_8:` 
	unknown_9: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallcategorycityprice_categoryunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2			*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_query_model_1_layer_call_and_return_conditional_losses_481064o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������:���������:���������: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name481119:&"
 
_user_specified_name481117:&"
 
_user_specified_name481115:


_output_shapes
: :&	"
 
_user_specified_name481111:&"
 
_user_specified_name481109:

_output_shapes
: :&"
 
_user_specified_name481105:&"
 
_user_specified_name481103:

_output_shapes
: :&"
 
_user_specified_name481099:SO
#
_output_shapes
:���������
(
_user_specified_namePrice_Category:IE
#
_output_shapes
:���������

_user_specified_nameCity:M I
#
_output_shapes
:���������
"
_user_specified_name
Category
�
�
G__inference_embedding_7_layer_call_and_return_conditional_losses_480826

inputs	)
embedding_lookup_480821: 
identity��embedding_lookup�
embedding_lookupResourceGatherembedding_lookup_480821inputs*
Tindices0	**
_class 
loc:@embedding_lookup/480821*'
_output_shapes
:��������� *
dtype0r
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*'
_output_shapes
:��������� q
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*'
_output_shapes
:��������� 5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup:&"
 
_user_specified_name480821:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
(__inference_dense_2_layer_call_fn_481343

inputs
unknown:` 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_480991o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������`: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name481339:&"
 
_user_specified_name481337:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�
�
H__inference_sequential_9_layer_call_and_return_conditional_losses_480831
string_lookup_7_input>
:string_lookup_7_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_7_none_lookup_lookuptablefindv2_default_value	$
embedding_7_480827: 
identity��#embedding_7/StatefulPartitionedCall�-string_lookup_7/None_Lookup/LookupTableFindV2�
-string_lookup_7/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_7_none_lookup_lookuptablefindv2_table_handlestring_lookup_7_input;string_lookup_7_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
string_lookup_7/IdentityIdentity6string_lookup_7/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
#embedding_7/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_7/Identity:output:0embedding_7_480827*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_embedding_7_layer_call_and_return_conditional_losses_480826{
IdentityIdentity,embedding_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� x
NoOpNoOp$^embedding_7/StatefulPartitionedCall.^string_lookup_7/None_Lookup/LookupTableFindV2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: : : 2J
#embedding_7/StatefulPartitionedCall#embedding_7/StatefulPartitionedCall2^
-string_lookup_7/None_Lookup/LookupTableFindV2-string_lookup_7/None_Lookup/LookupTableFindV2:&"
 
_user_specified_name480827:

_output_shapes
: :,(
&
_user_specified_nametable_handle:Z V
#
_output_shapes
:���������
/
_user_specified_namestring_lookup_7_input
�
�
G__inference_brute_force_layer_call_and_return_conditional_losses_481228
category
city
price_category
query_model_1_481195
query_model_1_481197	&
query_model_1_481199: 
query_model_1_481201
query_model_1_481203	&
query_model_1_481205: 
query_model_1_481207
query_model_1_481209	&
query_model_1_481211: &
query_model_1_481213:` "
query_model_1_481215: 1
matmul_readvariableop_resource:	�	 
gather_resource:	�	
identity

identity_1��Gather�MatMul/ReadVariableOp�%query_model_1/StatefulPartitionedCall�
%query_model_1/StatefulPartitionedCallStatefulPartitionedCallcategorycityprice_categoryquery_model_1_481195query_model_1_481197query_model_1_481199query_model_1_481201query_model_1_481203query_model_1_481205query_model_1_481207query_model_1_481209query_model_1_481211query_model_1_481213query_model_1_481215*
Tin
2			*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_query_model_1_layer_call_and_return_conditional_losses_481094u
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�	 *
dtype0�
MatMulMatMul.query_model_1/StatefulPartitionedCall:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������	*
transpose_b(J
TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :
z
TopKV2TopKV2MatMul:product:0TopKV2/k:output:0*
T0*:
_output_shapes(
&:���������
:���������
�
GatherResourceGathergather_resourceTopKV2:indices:0*
Tindices0*'
_output_shapes
:���������
*
dtype0^
IdentityIdentityTopKV2:values:0^NoOp*
T0*'
_output_shapes
:���������
`

Identity_1IdentityGather:output:0^NoOp*
T0*'
_output_shapes
:���������
k
NoOpNoOp^Gather^MatMul/ReadVariableOp&^query_model_1/StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������:���������:���������: : : : : : : : : : : : : 2
GatherGather2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2N
%query_model_1/StatefulPartitionedCall%query_model_1/StatefulPartitionedCall:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:&"
 
_user_specified_name481215:&"
 
_user_specified_name481213:&"
 
_user_specified_name481211:


_output_shapes
: :&	"
 
_user_specified_name481207:&"
 
_user_specified_name481205:

_output_shapes
: :&"
 
_user_specified_name481201:&"
 
_user_specified_name481199:

_output_shapes
: :&"
 
_user_specified_name481195:SO
#
_output_shapes
:���������
(
_user_specified_namePrice_Category:IE
#
_output_shapes
:���������

_user_specified_nameCity:M I
#
_output_shapes
:���������
"
_user_specified_name
Category
�
�
I__inference_query_model_1_layer_call_and_return_conditional_losses_481064
category
city
price_category
user_model_1_481039
user_model_1_481041	%
user_model_1_481043: 
user_model_1_481045
user_model_1_481047	%
user_model_1_481049: 
user_model_1_481051
user_model_1_481053	%
user_model_1_481055: &
sequential_10_481058:` "
sequential_10_481060: 
identity��%sequential_10/StatefulPartitionedCall�$user_model_1/StatefulPartitionedCall�
$user_model_1/StatefulPartitionedCallStatefulPartitionedCallcategorycityprice_categoryuser_model_1_481039user_model_1_481041user_model_1_481043user_model_1_481045user_model_1_481047user_model_1_481049user_model_1_481051user_model_1_481053user_model_1_481055*
Tin
2			*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_user_model_1_layer_call_and_return_conditional_losses_480900�
%sequential_10/StatefulPartitionedCallStatefulPartitionedCall-user_model_1/StatefulPartitionedCall:output:0sequential_10_481058sequential_10_481060*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_10_layer_call_and_return_conditional_losses_480998}
IdentityIdentity.sequential_10/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� q
NoOpNoOp&^sequential_10/StatefulPartitionedCall%^user_model_1/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������:���������:���������: : : : : : : : : : : 2N
%sequential_10/StatefulPartitionedCall%sequential_10/StatefulPartitionedCall2L
$user_model_1/StatefulPartitionedCall$user_model_1/StatefulPartitionedCall:&"
 
_user_specified_name481060:&"
 
_user_specified_name481058:&"
 
_user_specified_name481055:


_output_shapes
: :&	"
 
_user_specified_name481051:&"
 
_user_specified_name481049:

_output_shapes
: :&"
 
_user_specified_name481045:&"
 
_user_specified_name481043:

_output_shapes
: :&"
 
_user_specified_name481039:SO
#
_output_shapes
:���������
(
_user_specified_namePrice_Category:IE
#
_output_shapes
:���������

_user_specified_nameCity:M I
#
_output_shapes
:���������
"
_user_specified_name
Category
�
�
-__inference_sequential_8_layer_call_fn_480805
string_lookup_6_input
unknown
	unknown_0	
	unknown_1: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallstring_lookup_6_inputunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_8_layer_call_and_return_conditional_losses_480783o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name480801:

_output_shapes
: :&"
 
_user_specified_name480797:Z V
#
_output_shapes
:���������
/
_user_specified_namestring_lookup_6_input
�
�
,__inference_brute_force_layer_call_fn_481263
category
city
price_category
unknown
	unknown_0	
	unknown_1: 
	unknown_2
	unknown_3	
	unknown_4: 
	unknown_5
	unknown_6	
	unknown_7: 
	unknown_8:` 
	unknown_9: 

unknown_10:	�	 

unknown_11:	�	
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallcategorycityprice_categoryunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2			*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������
:���������
*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_brute_force_layer_call_and_return_conditional_losses_481190o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������:���������:���������: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name481257:&"
 
_user_specified_name481255:&"
 
_user_specified_name481253:&"
 
_user_specified_name481251:&"
 
_user_specified_name481249:


_output_shapes
: :&	"
 
_user_specified_name481245:&"
 
_user_specified_name481243:

_output_shapes
: :&"
 
_user_specified_name481239:&"
 
_user_specified_name481237:

_output_shapes
: :&"
 
_user_specified_name481233:SO
#
_output_shapes
:���������
(
_user_specified_namePrice_Category:IE
#
_output_shapes
:���������

_user_specified_nameCity:M I
#
_output_shapes
:���������
"
_user_specified_name
Category
�
�
G__inference_embedding_7_layer_call_and_return_conditional_losses_481398

inputs	)
embedding_lookup_481393: 
identity��embedding_lookup�
embedding_lookupResourceGatherembedding_lookup_481393inputs*
Tindices0	**
_class 
loc:@embedding_lookup/481393*'
_output_shapes
:��������� *
dtype0r
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*'
_output_shapes
:��������� q
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*'
_output_shapes
:��������� 5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup:&"
 
_user_specified_name481393:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
-__inference_sequential_7_layer_call_fn_480735
string_lookup_5_input
unknown
	unknown_0	
	unknown_1: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallstring_lookup_5_inputunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_7_layer_call_and_return_conditional_losses_480713o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name480731:

_output_shapes
: :&"
 
_user_specified_name480727:Z V
#
_output_shapes
:���������
/
_user_specified_namestring_lookup_5_input
�	
�
C__inference_dense_2_layer_call_and_return_conditional_losses_480991

inputs0
matmul_readvariableop_resource:` -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:` *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:��������� S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�
�
H__inference_user_model_1_layer_call_and_return_conditional_losses_480929
category
city
price_category
sequential_7_480905
sequential_7_480907	%
sequential_7_480909: 
sequential_8_480912
sequential_8_480914	%
sequential_8_480916: 
sequential_9_480919
sequential_9_480921	%
sequential_9_480923: 
identity��$sequential_7/StatefulPartitionedCall�$sequential_8/StatefulPartitionedCall�$sequential_9/StatefulPartitionedCall�
$sequential_7/StatefulPartitionedCallStatefulPartitionedCallcitysequential_7_480905sequential_7_480907sequential_7_480909*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_7_layer_call_and_return_conditional_losses_480724�
$sequential_8/StatefulPartitionedCallStatefulPartitionedCallcategorysequential_8_480912sequential_8_480914sequential_8_480916*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_8_layer_call_and_return_conditional_losses_480783�
$sequential_9/StatefulPartitionedCallStatefulPartitionedCallprice_categorysequential_9_480919sequential_9_480921sequential_9_480923*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_9_layer_call_and_return_conditional_losses_480842M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2-sequential_7/StatefulPartitionedCall:output:0-sequential_8/StatefulPartitionedCall:output:0-sequential_9/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:���������`^
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:���������`�
NoOpNoOp%^sequential_7/StatefulPartitionedCall%^sequential_8/StatefulPartitionedCall%^sequential_9/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������:���������: : : : : : : : : 2L
$sequential_7/StatefulPartitionedCall$sequential_7/StatefulPartitionedCall2L
$sequential_8/StatefulPartitionedCall$sequential_8/StatefulPartitionedCall2L
$sequential_9/StatefulPartitionedCall$sequential_9/StatefulPartitionedCall:&"
 
_user_specified_name480923:


_output_shapes
: :&	"
 
_user_specified_name480919:&"
 
_user_specified_name480916:

_output_shapes
: :&"
 
_user_specified_name480912:&"
 
_user_specified_name480909:

_output_shapes
: :&"
 
_user_specified_name480905:SO
#
_output_shapes
:���������
(
_user_specified_namePrice_Category:IE
#
_output_shapes
:���������

_user_specified_nameCity:M I
#
_output_shapes
:���������
"
_user_specified_name
Category
�
�
$__inference_signature_wrapper_481334
category
city
price_category
unknown
	unknown_0	
	unknown_1: 
	unknown_2
	unknown_3	
	unknown_4: 
	unknown_5
	unknown_6	
	unknown_7: 
	unknown_8:` 
	unknown_9: 

unknown_10:	�	 

unknown_11:	�	
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallcategorycityprice_categoryunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2			*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������
:���������
*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_480694o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������:���������:���������: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name481328:&"
 
_user_specified_name481326:&"
 
_user_specified_name481324:&"
 
_user_specified_name481322:&"
 
_user_specified_name481320:


_output_shapes
: :&	"
 
_user_specified_name481316:&"
 
_user_specified_name481314:

_output_shapes
: :&"
 
_user_specified_name481310:&"
 
_user_specified_name481308:

_output_shapes
: :&"
 
_user_specified_name481304:SO
#
_output_shapes
:���������
(
_user_specified_namePrice_Category:IE
#
_output_shapes
:���������

_user_specified_nameCity:M I
#
_output_shapes
:���������
"
_user_specified_name
Category
�
�
G__inference_embedding_6_layer_call_and_return_conditional_losses_481383

inputs	)
embedding_lookup_481378: 
identity��embedding_lookup�
embedding_lookupResourceGatherembedding_lookup_481378inputs*
Tindices0	**
_class 
loc:@embedding_lookup/481378*'
_output_shapes
:��������� *
dtype0r
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*'
_output_shapes
:��������� q
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*'
_output_shapes
:��������� 5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup:&"
 
_user_specified_name481378:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
__inference__initializer_4814099
5key_value_init101384_lookuptableimportv2_table_handle1
-key_value_init101384_lookuptableimportv2_keys3
/key_value_init101384_lookuptableimportv2_values	
identity��(key_value_init101384/LookupTableImportV2�
(key_value_init101384/LookupTableImportV2LookupTableImportV25key_value_init101384_lookuptableimportv2_table_handle-key_value_init101384_lookuptableimportv2_keys/key_value_init101384_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: M
NoOpNoOp)^key_value_init101384/LookupTableImportV2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2T
(key_value_init101384/LookupTableImportV2(key_value_init101384/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
::, (
&
_user_specified_nametable_handle
�
�
,__inference_embedding_6_layer_call_fn_481375

inputs	
unknown: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_embedding_6_layer_call_and_return_conditional_losses_480767o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:���������: 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name481371:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
,__inference_embedding_5_layer_call_fn_481360

inputs	
unknown: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_embedding_5_layer_call_and_return_conditional_losses_480708o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:���������: 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name481356:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�%
�
"__inference__traced_restore_481555
file_prefix+
assignvariableop_identifiers:	�	0
assignvariableop_1_candidates:	�	 ;
)assignvariableop_2_embedding_5_embeddings: ;
)assignvariableop_3_embedding_6_embeddings: ;
)assignvariableop_4_embedding_7_embeddings: 3
!assignvariableop_5_dense_2_kernel:` -
assignvariableop_6_dense_2_bias: 

identity_8��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&identifiers/.ATTRIBUTES/VARIABLE_VALUEB%candidates/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*4
_output_shapes"
 ::::::::*
dtypes

2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_identifiersIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_candidatesIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp)assignvariableop_2_embedding_5_embeddingsIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp)assignvariableop_3_embedding_6_embeddingsIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp)assignvariableop_4_embedding_7_embeddingsIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_2_kernelIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_2_biasIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �

Identity_7Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_8IdentityIdentity_7:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6*
_output_shapes
 "!

identity_8Identity_8:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: : : : : : : : 2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62$
AssignVariableOpAssignVariableOp:,(
&
_user_specified_namedense_2/bias:.*
(
_user_specified_namedense_2/kernel:62
0
_user_specified_nameembedding_7/embeddings:62
0
_user_specified_nameembedding_6/embeddings:62
0
_user_specified_nameembedding_5/embeddings:*&
$
_user_specified_name
candidates:+'
%
_user_specified_nameidentifiers:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
I__inference_query_model_1_layer_call_and_return_conditional_losses_481094
category
city
price_category
user_model_1_481069
user_model_1_481071	%
user_model_1_481073: 
user_model_1_481075
user_model_1_481077	%
user_model_1_481079: 
user_model_1_481081
user_model_1_481083	%
user_model_1_481085: &
sequential_10_481088:` "
sequential_10_481090: 
identity��%sequential_10/StatefulPartitionedCall�$user_model_1/StatefulPartitionedCall�
$user_model_1/StatefulPartitionedCallStatefulPartitionedCallcategorycityprice_categoryuser_model_1_481069user_model_1_481071user_model_1_481073user_model_1_481075user_model_1_481077user_model_1_481079user_model_1_481081user_model_1_481083user_model_1_481085*
Tin
2			*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_user_model_1_layer_call_and_return_conditional_losses_480929�
%sequential_10/StatefulPartitionedCallStatefulPartitionedCall-user_model_1/StatefulPartitionedCall:output:0sequential_10_481088sequential_10_481090*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_10_layer_call_and_return_conditional_losses_481007}
IdentityIdentity.sequential_10/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� q
NoOpNoOp&^sequential_10/StatefulPartitionedCall%^user_model_1/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������:���������:���������: : : : : : : : : : : 2N
%sequential_10/StatefulPartitionedCall%sequential_10/StatefulPartitionedCall2L
$user_model_1/StatefulPartitionedCall$user_model_1/StatefulPartitionedCall:&"
 
_user_specified_name481090:&"
 
_user_specified_name481088:&"
 
_user_specified_name481085:


_output_shapes
: :&	"
 
_user_specified_name481081:&"
 
_user_specified_name481079:

_output_shapes
: :&"
 
_user_specified_name481075:&"
 
_user_specified_name481073:

_output_shapes
: :&"
 
_user_specified_name481069:SO
#
_output_shapes
:���������
(
_user_specified_namePrice_Category:IE
#
_output_shapes
:���������

_user_specified_nameCity:M I
#
_output_shapes
:���������
"
_user_specified_name
Category
�
�
G__inference_embedding_5_layer_call_and_return_conditional_losses_480708

inputs	)
embedding_lookup_480703: 
identity��embedding_lookup�
embedding_lookupResourceGatherembedding_lookup_480703inputs*
Tindices0	**
_class 
loc:@embedding_lookup/480703*'
_output_shapes
:��������� *
dtype0r
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*'
_output_shapes
:��������� q
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*'
_output_shapes
:��������� 5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup:&"
 
_user_specified_name480703:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
-__inference_user_model_1_layer_call_fn_480979
category
city
price_category
unknown
	unknown_0	
	unknown_1: 
	unknown_2
	unknown_3	
	unknown_4: 
	unknown_5
	unknown_6	
	unknown_7: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallcategorycityprice_categoryunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2			*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_user_model_1_layer_call_and_return_conditional_losses_480929o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������:���������: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name480975:


_output_shapes
: :&	"
 
_user_specified_name480971:&"
 
_user_specified_name480969:

_output_shapes
: :&"
 
_user_specified_name480965:&"
 
_user_specified_name480963:

_output_shapes
: :&"
 
_user_specified_name480959:SO
#
_output_shapes
:���������
(
_user_specified_namePrice_Category:IE
#
_output_shapes
:���������

_user_specified_nameCity:M I
#
_output_shapes
:���������
"
_user_specified_name
Category
�
�
,__inference_embedding_7_layer_call_fn_481390

inputs	
unknown: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_embedding_7_layer_call_and_return_conditional_losses_480826o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:���������: 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name481386:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
-__inference_sequential_9_layer_call_fn_480864
string_lookup_7_input
unknown
	unknown_0	
	unknown_1: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallstring_lookup_7_inputunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_9_layer_call_and_return_conditional_losses_480842o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name480860:

_output_shapes
: :&"
 
_user_specified_name480856:Z V
#
_output_shapes
:���������
/
_user_specified_namestring_lookup_7_input
�
�
G__inference_brute_force_layer_call_and_return_conditional_losses_481190
category
city
price_category
query_model_1_481157
query_model_1_481159	&
query_model_1_481161: 
query_model_1_481163
query_model_1_481165	&
query_model_1_481167: 
query_model_1_481169
query_model_1_481171	&
query_model_1_481173: &
query_model_1_481175:` "
query_model_1_481177: 1
matmul_readvariableop_resource:	�	 
gather_resource:	�	
identity

identity_1��Gather�MatMul/ReadVariableOp�%query_model_1/StatefulPartitionedCall�
%query_model_1/StatefulPartitionedCallStatefulPartitionedCallcategorycityprice_categoryquery_model_1_481157query_model_1_481159query_model_1_481161query_model_1_481163query_model_1_481165query_model_1_481167query_model_1_481169query_model_1_481171query_model_1_481173query_model_1_481175query_model_1_481177*
Tin
2			*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_query_model_1_layer_call_and_return_conditional_losses_481064u
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�	 *
dtype0�
MatMulMatMul.query_model_1/StatefulPartitionedCall:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������	*
transpose_b(J
TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :
z
TopKV2TopKV2MatMul:product:0TopKV2/k:output:0*
T0*:
_output_shapes(
&:���������
:���������
�
GatherResourceGathergather_resourceTopKV2:indices:0*
Tindices0*'
_output_shapes
:���������
*
dtype0^
IdentityIdentityTopKV2:values:0^NoOp*
T0*'
_output_shapes
:���������
`

Identity_1IdentityGather:output:0^NoOp*
T0*'
_output_shapes
:���������
k
NoOpNoOp^Gather^MatMul/ReadVariableOp&^query_model_1/StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������:���������:���������: : : : : : : : : : : : : 2
GatherGather2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2N
%query_model_1/StatefulPartitionedCall%query_model_1/StatefulPartitionedCall:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:&"
 
_user_specified_name481177:&"
 
_user_specified_name481175:&"
 
_user_specified_name481173:


_output_shapes
: :&	"
 
_user_specified_name481169:&"
 
_user_specified_name481167:

_output_shapes
: :&"
 
_user_specified_name481163:&"
 
_user_specified_name481161:

_output_shapes
: :&"
 
_user_specified_name481157:SO
#
_output_shapes
:���������
(
_user_specified_namePrice_Category:IE
#
_output_shapes
:���������

_user_specified_nameCity:M I
#
_output_shapes
:���������
"
_user_specified_name
Category
�
-
__inference__destroyer_481413
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�
;
__inference__creator_481432
identity��
hash_tablen

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name101431*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: /
NoOpNoOp^hash_table*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
�
;
__inference__creator_481417
identity��
hash_tablen

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name101408*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: /
NoOpNoOp^hash_table*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
�
�
H__inference_sequential_8_layer_call_and_return_conditional_losses_480783
string_lookup_6_input>
:string_lookup_6_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_6_none_lookup_lookuptablefindv2_default_value	$
embedding_6_480779: 
identity��#embedding_6/StatefulPartitionedCall�-string_lookup_6/None_Lookup/LookupTableFindV2�
-string_lookup_6/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_6_none_lookup_lookuptablefindv2_table_handlestring_lookup_6_input;string_lookup_6_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
string_lookup_6/IdentityIdentity6string_lookup_6/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
#embedding_6/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_6/Identity:output:0embedding_6_480779*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_embedding_6_layer_call_and_return_conditional_losses_480767{
IdentityIdentity,embedding_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� x
NoOpNoOp$^embedding_6/StatefulPartitionedCall.^string_lookup_6/None_Lookup/LookupTableFindV2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: : : 2J
#embedding_6/StatefulPartitionedCall#embedding_6/StatefulPartitionedCall2^
-string_lookup_6/None_Lookup/LookupTableFindV2-string_lookup_6/None_Lookup/LookupTableFindV2:&"
 
_user_specified_name480779:

_output_shapes
: :,(
&
_user_specified_nametable_handle:Z V
#
_output_shapes
:���������
/
_user_specified_namestring_lookup_6_input
�
�
__inference__initializer_4814399
5key_value_init101430_lookuptableimportv2_table_handle1
-key_value_init101430_lookuptableimportv2_keys3
/key_value_init101430_lookuptableimportv2_values	
identity��(key_value_init101430/LookupTableImportV2�
(key_value_init101430/LookupTableImportV2LookupTableImportV25key_value_init101430_lookuptableimportv2_table_handle-key_value_init101430_lookuptableimportv2_keys/key_value_init101430_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: M
NoOpNoOp)^key_value_init101430/LookupTableImportV2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2T
(key_value_init101430/LookupTableImportV2(key_value_init101430/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
::, (
&
_user_specified_nametable_handle
�
�
I__inference_sequential_10_layer_call_and_return_conditional_losses_480998
dense_2_input 
dense_2_480992:` 
dense_2_480994: 
identity��dense_2/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCalldense_2_inputdense_2_480992dense_2_480994*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_480991w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� D
NoOpNoOp ^dense_2/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������`: : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:&"
 
_user_specified_name480994:&"
 
_user_specified_name480992:V R
'
_output_shapes
:���������`
'
_user_specified_namedense_2_input
�
-
__inference__destroyer_481443
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�
�
H__inference_sequential_9_layer_call_and_return_conditional_losses_480842
string_lookup_7_input>
:string_lookup_7_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_7_none_lookup_lookuptablefindv2_default_value	$
embedding_7_480838: 
identity��#embedding_7/StatefulPartitionedCall�-string_lookup_7/None_Lookup/LookupTableFindV2�
-string_lookup_7/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_7_none_lookup_lookuptablefindv2_table_handlestring_lookup_7_input;string_lookup_7_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
string_lookup_7/IdentityIdentity6string_lookup_7/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
#embedding_7/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_7/Identity:output:0embedding_7_480838*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_embedding_7_layer_call_and_return_conditional_losses_480826{
IdentityIdentity,embedding_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� x
NoOpNoOp$^embedding_7/StatefulPartitionedCall.^string_lookup_7/None_Lookup/LookupTableFindV2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: : : 2J
#embedding_7/StatefulPartitionedCall#embedding_7/StatefulPartitionedCall2^
-string_lookup_7/None_Lookup/LookupTableFindV2-string_lookup_7/None_Lookup/LookupTableFindV2:&"
 
_user_specified_name480838:

_output_shapes
: :,(
&
_user_specified_nametable_handle:Z V
#
_output_shapes
:���������
/
_user_specified_namestring_lookup_7_input
�
�
.__inference_sequential_10_layer_call_fn_481025
dense_2_input
unknown:` 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_2_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_10_layer_call_and_return_conditional_losses_481007o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������`: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name481021:&"
 
_user_specified_name481019:V R
'
_output_shapes
:���������`
'
_user_specified_namedense_2_input
�
-
__inference__destroyer_481428
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�^
�
!__inference__wrapped_model_480694
category
city
price_categoryr
nbrute_force_query_model_1_user_model_1_sequential_7_string_lookup_5_none_lookup_lookuptablefindv2_table_handles
obrute_force_query_model_1_user_model_1_sequential_7_string_lookup_5_none_lookup_lookuptablefindv2_default_value	i
Wbrute_force_query_model_1_user_model_1_sequential_7_embedding_5_embedding_lookup_480656: r
nbrute_force_query_model_1_user_model_1_sequential_8_string_lookup_6_none_lookup_lookuptablefindv2_table_handles
obrute_force_query_model_1_user_model_1_sequential_8_string_lookup_6_none_lookup_lookuptablefindv2_default_value	i
Wbrute_force_query_model_1_user_model_1_sequential_8_embedding_6_embedding_lookup_480664: r
nbrute_force_query_model_1_user_model_1_sequential_9_string_lookup_7_none_lookup_lookuptablefindv2_table_handles
obrute_force_query_model_1_user_model_1_sequential_9_string_lookup_7_none_lookup_lookuptablefindv2_default_value	i
Wbrute_force_query_model_1_user_model_1_sequential_9_embedding_7_embedding_lookup_480672: `
Nbrute_force_query_model_1_sequential_10_dense_2_matmul_readvariableop_resource:` ]
Obrute_force_query_model_1_sequential_10_dense_2_biasadd_readvariableop_resource: =
*brute_force_matmul_readvariableop_resource:	�	 *
brute_force_gather_resource:	�	
identity

identity_1��brute_force/Gather�!brute_force/MatMul/ReadVariableOp�Fbrute_force/query_model_1/sequential_10/dense_2/BiasAdd/ReadVariableOp�Ebrute_force/query_model_1/sequential_10/dense_2/MatMul/ReadVariableOp�Pbrute_force/query_model_1/user_model_1/sequential_7/embedding_5/embedding_lookup�abrute_force/query_model_1/user_model_1/sequential_7/string_lookup_5/None_Lookup/LookupTableFindV2�Pbrute_force/query_model_1/user_model_1/sequential_8/embedding_6/embedding_lookup�abrute_force/query_model_1/user_model_1/sequential_8/string_lookup_6/None_Lookup/LookupTableFindV2�Pbrute_force/query_model_1/user_model_1/sequential_9/embedding_7/embedding_lookup�abrute_force/query_model_1/user_model_1/sequential_9/string_lookup_7/None_Lookup/LookupTableFindV2�
abrute_force/query_model_1/user_model_1/sequential_7/string_lookup_5/None_Lookup/LookupTableFindV2LookupTableFindV2nbrute_force_query_model_1_user_model_1_sequential_7_string_lookup_5_none_lookup_lookuptablefindv2_table_handlecityobrute_force_query_model_1_user_model_1_sequential_7_string_lookup_5_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
Lbrute_force/query_model_1/user_model_1/sequential_7/string_lookup_5/IdentityIdentityjbrute_force/query_model_1/user_model_1/sequential_7/string_lookup_5/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
Pbrute_force/query_model_1/user_model_1/sequential_7/embedding_5/embedding_lookupResourceGatherWbrute_force_query_model_1_user_model_1_sequential_7_embedding_5_embedding_lookup_480656Ubrute_force/query_model_1/user_model_1/sequential_7/string_lookup_5/Identity:output:0*
Tindices0	*j
_class`
^\loc:@brute_force/query_model_1/user_model_1/sequential_7/embedding_5/embedding_lookup/480656*'
_output_shapes
:��������� *
dtype0�
Ybrute_force/query_model_1/user_model_1/sequential_7/embedding_5/embedding_lookup/IdentityIdentityYbrute_force/query_model_1/user_model_1/sequential_7/embedding_5/embedding_lookup:output:0*
T0*'
_output_shapes
:��������� �
abrute_force/query_model_1/user_model_1/sequential_8/string_lookup_6/None_Lookup/LookupTableFindV2LookupTableFindV2nbrute_force_query_model_1_user_model_1_sequential_8_string_lookup_6_none_lookup_lookuptablefindv2_table_handlecategoryobrute_force_query_model_1_user_model_1_sequential_8_string_lookup_6_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
Lbrute_force/query_model_1/user_model_1/sequential_8/string_lookup_6/IdentityIdentityjbrute_force/query_model_1/user_model_1/sequential_8/string_lookup_6/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
Pbrute_force/query_model_1/user_model_1/sequential_8/embedding_6/embedding_lookupResourceGatherWbrute_force_query_model_1_user_model_1_sequential_8_embedding_6_embedding_lookup_480664Ubrute_force/query_model_1/user_model_1/sequential_8/string_lookup_6/Identity:output:0*
Tindices0	*j
_class`
^\loc:@brute_force/query_model_1/user_model_1/sequential_8/embedding_6/embedding_lookup/480664*'
_output_shapes
:��������� *
dtype0�
Ybrute_force/query_model_1/user_model_1/sequential_8/embedding_6/embedding_lookup/IdentityIdentityYbrute_force/query_model_1/user_model_1/sequential_8/embedding_6/embedding_lookup:output:0*
T0*'
_output_shapes
:��������� �
abrute_force/query_model_1/user_model_1/sequential_9/string_lookup_7/None_Lookup/LookupTableFindV2LookupTableFindV2nbrute_force_query_model_1_user_model_1_sequential_9_string_lookup_7_none_lookup_lookuptablefindv2_table_handleprice_categoryobrute_force_query_model_1_user_model_1_sequential_9_string_lookup_7_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
Lbrute_force/query_model_1/user_model_1/sequential_9/string_lookup_7/IdentityIdentityjbrute_force/query_model_1/user_model_1/sequential_9/string_lookup_7/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
Pbrute_force/query_model_1/user_model_1/sequential_9/embedding_7/embedding_lookupResourceGatherWbrute_force_query_model_1_user_model_1_sequential_9_embedding_7_embedding_lookup_480672Ubrute_force/query_model_1/user_model_1/sequential_9/string_lookup_7/Identity:output:0*
Tindices0	*j
_class`
^\loc:@brute_force/query_model_1/user_model_1/sequential_9/embedding_7/embedding_lookup/480672*'
_output_shapes
:��������� *
dtype0�
Ybrute_force/query_model_1/user_model_1/sequential_9/embedding_7/embedding_lookup/IdentityIdentityYbrute_force/query_model_1/user_model_1/sequential_9/embedding_7/embedding_lookup:output:0*
T0*'
_output_shapes
:��������� t
2brute_force/query_model_1/user_model_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
-brute_force/query_model_1/user_model_1/concatConcatV2bbrute_force/query_model_1/user_model_1/sequential_7/embedding_5/embedding_lookup/Identity:output:0bbrute_force/query_model_1/user_model_1/sequential_8/embedding_6/embedding_lookup/Identity:output:0bbrute_force/query_model_1/user_model_1/sequential_9/embedding_7/embedding_lookup/Identity:output:0;brute_force/query_model_1/user_model_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������`�
Ebrute_force/query_model_1/sequential_10/dense_2/MatMul/ReadVariableOpReadVariableOpNbrute_force_query_model_1_sequential_10_dense_2_matmul_readvariableop_resource*
_output_shapes

:` *
dtype0�
6brute_force/query_model_1/sequential_10/dense_2/MatMulMatMul6brute_force/query_model_1/user_model_1/concat:output:0Mbrute_force/query_model_1/sequential_10/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
Fbrute_force/query_model_1/sequential_10/dense_2/BiasAdd/ReadVariableOpReadVariableOpObrute_force_query_model_1_sequential_10_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
7brute_force/query_model_1/sequential_10/dense_2/BiasAddBiasAdd@brute_force/query_model_1/sequential_10/dense_2/MatMul:product:0Nbrute_force/query_model_1/sequential_10/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
!brute_force/MatMul/ReadVariableOpReadVariableOp*brute_force_matmul_readvariableop_resource*
_output_shapes
:	�	 *
dtype0�
brute_force/MatMulMatMul@brute_force/query_model_1/sequential_10/dense_2/BiasAdd:output:0)brute_force/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������	*
transpose_b(V
brute_force/TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :
�
brute_force/TopKV2TopKV2brute_force/MatMul:product:0brute_force/TopKV2/k:output:0*
T0*:
_output_shapes(
&:���������
:���������
�
brute_force/GatherResourceGatherbrute_force_gather_resourcebrute_force/TopKV2:indices:0*
Tindices0*'
_output_shapes
:���������
*
dtype0j
IdentityIdentitybrute_force/TopKV2:values:0^NoOp*
T0*'
_output_shapes
:���������
l

Identity_1Identitybrute_force/Gather:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^brute_force/Gather"^brute_force/MatMul/ReadVariableOpG^brute_force/query_model_1/sequential_10/dense_2/BiasAdd/ReadVariableOpF^brute_force/query_model_1/sequential_10/dense_2/MatMul/ReadVariableOpQ^brute_force/query_model_1/user_model_1/sequential_7/embedding_5/embedding_lookupb^brute_force/query_model_1/user_model_1/sequential_7/string_lookup_5/None_Lookup/LookupTableFindV2Q^brute_force/query_model_1/user_model_1/sequential_8/embedding_6/embedding_lookupb^brute_force/query_model_1/user_model_1/sequential_8/string_lookup_6/None_Lookup/LookupTableFindV2Q^brute_force/query_model_1/user_model_1/sequential_9/embedding_7/embedding_lookupb^brute_force/query_model_1/user_model_1/sequential_9/string_lookup_7/None_Lookup/LookupTableFindV2*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������:���������:���������: : : : : : : : : : : : : 2(
brute_force/Gatherbrute_force/Gather2F
!brute_force/MatMul/ReadVariableOp!brute_force/MatMul/ReadVariableOp2�
Fbrute_force/query_model_1/sequential_10/dense_2/BiasAdd/ReadVariableOpFbrute_force/query_model_1/sequential_10/dense_2/BiasAdd/ReadVariableOp2�
Ebrute_force/query_model_1/sequential_10/dense_2/MatMul/ReadVariableOpEbrute_force/query_model_1/sequential_10/dense_2/MatMul/ReadVariableOp2�
Pbrute_force/query_model_1/user_model_1/sequential_7/embedding_5/embedding_lookupPbrute_force/query_model_1/user_model_1/sequential_7/embedding_5/embedding_lookup2�
abrute_force/query_model_1/user_model_1/sequential_7/string_lookup_5/None_Lookup/LookupTableFindV2abrute_force/query_model_1/user_model_1/sequential_7/string_lookup_5/None_Lookup/LookupTableFindV22�
Pbrute_force/query_model_1/user_model_1/sequential_8/embedding_6/embedding_lookupPbrute_force/query_model_1/user_model_1/sequential_8/embedding_6/embedding_lookup2�
abrute_force/query_model_1/user_model_1/sequential_8/string_lookup_6/None_Lookup/LookupTableFindV2abrute_force/query_model_1/user_model_1/sequential_8/string_lookup_6/None_Lookup/LookupTableFindV22�
Pbrute_force/query_model_1/user_model_1/sequential_9/embedding_7/embedding_lookupPbrute_force/query_model_1/user_model_1/sequential_9/embedding_7/embedding_lookup2�
abrute_force/query_model_1/user_model_1/sequential_9/string_lookup_7/None_Lookup/LookupTableFindV2abrute_force/query_model_1/user_model_1/sequential_9/string_lookup_7/None_Lookup/LookupTableFindV2:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:&"
 
_user_specified_name480672:


_output_shapes
: :,	(
&
_user_specified_nametable_handle:&"
 
_user_specified_name480664:

_output_shapes
: :,(
&
_user_specified_nametable_handle:&"
 
_user_specified_name480656:

_output_shapes
: :,(
&
_user_specified_nametable_handle:SO
#
_output_shapes
:���������
(
_user_specified_namePrice_Category:IE
#
_output_shapes
:���������

_user_specified_nameCity:M I
#
_output_shapes
:���������
"
_user_specified_name
Category
�
�
__inference__initializer_4814249
5key_value_init101407_lookuptableimportv2_table_handle1
-key_value_init101407_lookuptableimportv2_keys3
/key_value_init101407_lookuptableimportv2_values	
identity��(key_value_init101407/LookupTableImportV2�
(key_value_init101407/LookupTableImportV2LookupTableImportV25key_value_init101407_lookuptableimportv2_table_handle-key_value_init101407_lookuptableimportv2_keys/key_value_init101407_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: M
NoOpNoOp)^key_value_init101407/LookupTableImportV2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2T
(key_value_init101407/LookupTableImportV2(key_value_init101407/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
::, (
&
_user_specified_nametable_handle
�
�
-__inference_sequential_7_layer_call_fn_480746
string_lookup_5_input
unknown
	unknown_0	
	unknown_1: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallstring_lookup_5_inputunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_7_layer_call_and_return_conditional_losses_480724o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name480742:

_output_shapes
: :&"
 
_user_specified_name480738:Z V
#
_output_shapes
:���������
/
_user_specified_namestring_lookup_5_input
�
�
-__inference_sequential_9_layer_call_fn_480853
string_lookup_7_input
unknown
	unknown_0	
	unknown_1: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallstring_lookup_7_inputunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_9_layer_call_and_return_conditional_losses_480831o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name480849:

_output_shapes
: :&"
 
_user_specified_name480845:Z V
#
_output_shapes
:���������
/
_user_specified_namestring_lookup_7_input
�
�
G__inference_embedding_5_layer_call_and_return_conditional_losses_481368

inputs	)
embedding_lookup_481363: 
identity��embedding_lookup�
embedding_lookupResourceGatherembedding_lookup_481363inputs*
Tindices0	**
_class 
loc:@embedding_lookup/481363*'
_output_shapes
:��������� *
dtype0r
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*'
_output_shapes
:��������� q
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*'
_output_shapes
:��������� 5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup:&"
 
_user_specified_name481363:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
H__inference_user_model_1_layer_call_and_return_conditional_losses_480900
category
city
price_category
sequential_7_480876
sequential_7_480878	%
sequential_7_480880: 
sequential_8_480883
sequential_8_480885	%
sequential_8_480887: 
sequential_9_480890
sequential_9_480892	%
sequential_9_480894: 
identity��$sequential_7/StatefulPartitionedCall�$sequential_8/StatefulPartitionedCall�$sequential_9/StatefulPartitionedCall�
$sequential_7/StatefulPartitionedCallStatefulPartitionedCallcitysequential_7_480876sequential_7_480878sequential_7_480880*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_7_layer_call_and_return_conditional_losses_480713�
$sequential_8/StatefulPartitionedCallStatefulPartitionedCallcategorysequential_8_480883sequential_8_480885sequential_8_480887*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_8_layer_call_and_return_conditional_losses_480772�
$sequential_9/StatefulPartitionedCallStatefulPartitionedCallprice_categorysequential_9_480890sequential_9_480892sequential_9_480894*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_9_layer_call_and_return_conditional_losses_480831M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2-sequential_7/StatefulPartitionedCall:output:0-sequential_8/StatefulPartitionedCall:output:0-sequential_9/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:���������`^
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:���������`�
NoOpNoOp%^sequential_7/StatefulPartitionedCall%^sequential_8/StatefulPartitionedCall%^sequential_9/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������:���������: : : : : : : : : 2L
$sequential_7/StatefulPartitionedCall$sequential_7/StatefulPartitionedCall2L
$sequential_8/StatefulPartitionedCall$sequential_8/StatefulPartitionedCall2L
$sequential_9/StatefulPartitionedCall$sequential_9/StatefulPartitionedCall:&"
 
_user_specified_name480894:


_output_shapes
: :&	"
 
_user_specified_name480890:&"
 
_user_specified_name480887:

_output_shapes
: :&"
 
_user_specified_name480883:&"
 
_user_specified_name480880:

_output_shapes
: :&"
 
_user_specified_name480876:SO
#
_output_shapes
:���������
(
_user_specified_namePrice_Category:IE
#
_output_shapes
:���������

_user_specified_nameCity:M I
#
_output_shapes
:���������
"
_user_specified_name
Category
�
�
G__inference_embedding_6_layer_call_and_return_conditional_losses_480767

inputs	)
embedding_lookup_480762: 
identity��embedding_lookup�
embedding_lookupResourceGatherembedding_lookup_480762inputs*
Tindices0	**
_class 
loc:@embedding_lookup/480762*'
_output_shapes
:��������� *
dtype0r
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*'
_output_shapes
:��������� q
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*'
_output_shapes
:��������� 5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup:&"
 
_user_specified_name480762:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
-__inference_user_model_1_layer_call_fn_480954
category
city
price_category
unknown
	unknown_0	
	unknown_1: 
	unknown_2
	unknown_3	
	unknown_4: 
	unknown_5
	unknown_6	
	unknown_7: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallcategorycityprice_categoryunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2			*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_user_model_1_layer_call_and_return_conditional_losses_480900o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������:���������: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name480950:


_output_shapes
: :&	"
 
_user_specified_name480946:&"
 
_user_specified_name480944:

_output_shapes
: :&"
 
_user_specified_name480940:&"
 
_user_specified_name480938:

_output_shapes
: :&"
 
_user_specified_name480934:SO
#
_output_shapes
:���������
(
_user_specified_namePrice_Category:IE
#
_output_shapes
:���������

_user_specified_nameCity:M I
#
_output_shapes
:���������
"
_user_specified_name
Category
�
�
H__inference_sequential_8_layer_call_and_return_conditional_losses_480772
string_lookup_6_input>
:string_lookup_6_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_6_none_lookup_lookuptablefindv2_default_value	$
embedding_6_480768: 
identity��#embedding_6/StatefulPartitionedCall�-string_lookup_6/None_Lookup/LookupTableFindV2�
-string_lookup_6/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_6_none_lookup_lookuptablefindv2_table_handlestring_lookup_6_input;string_lookup_6_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
string_lookup_6/IdentityIdentity6string_lookup_6/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
#embedding_6/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_6/Identity:output:0embedding_6_480768*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_embedding_6_layer_call_and_return_conditional_losses_480767{
IdentityIdentity,embedding_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� x
NoOpNoOp$^embedding_6/StatefulPartitionedCall.^string_lookup_6/None_Lookup/LookupTableFindV2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: : : 2J
#embedding_6/StatefulPartitionedCall#embedding_6/StatefulPartitionedCall2^
-string_lookup_6/None_Lookup/LookupTableFindV2-string_lookup_6/None_Lookup/LookupTableFindV2:&"
 
_user_specified_name480768:

_output_shapes
: :,(
&
_user_specified_nametable_handle:Z V
#
_output_shapes
:���������
/
_user_specified_namestring_lookup_6_input
�
�
H__inference_sequential_7_layer_call_and_return_conditional_losses_480713
string_lookup_5_input>
:string_lookup_5_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_5_none_lookup_lookuptablefindv2_default_value	$
embedding_5_480709: 
identity��#embedding_5/StatefulPartitionedCall�-string_lookup_5/None_Lookup/LookupTableFindV2�
-string_lookup_5/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_5_none_lookup_lookuptablefindv2_table_handlestring_lookup_5_input;string_lookup_5_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
string_lookup_5/IdentityIdentity6string_lookup_5/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
#embedding_5/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_5/Identity:output:0embedding_5_480709*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_embedding_5_layer_call_and_return_conditional_losses_480708{
IdentityIdentity,embedding_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� x
NoOpNoOp$^embedding_5/StatefulPartitionedCall.^string_lookup_5/None_Lookup/LookupTableFindV2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: : : 2J
#embedding_5/StatefulPartitionedCall#embedding_5/StatefulPartitionedCall2^
-string_lookup_5/None_Lookup/LookupTableFindV2-string_lookup_5/None_Lookup/LookupTableFindV2:&"
 
_user_specified_name480709:

_output_shapes
: :,(
&
_user_specified_nametable_handle:Z V
#
_output_shapes
:���������
/
_user_specified_namestring_lookup_5_input
�
�
.__inference_sequential_10_layer_call_fn_481016
dense_2_input
unknown:` 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_2_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_10_layer_call_and_return_conditional_losses_480998o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������`: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name481012:&"
 
_user_specified_name481010:V R
'
_output_shapes
:���������`
'
_user_specified_namedense_2_input
�
�
,__inference_brute_force_layer_call_fn_481298
category
city
price_category
unknown
	unknown_0	
	unknown_1: 
	unknown_2
	unknown_3	
	unknown_4: 
	unknown_5
	unknown_6	
	unknown_7: 
	unknown_8:` 
	unknown_9: 

unknown_10:	�	 

unknown_11:	�	
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallcategorycityprice_categoryunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2			*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������
:���������
*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_brute_force_layer_call_and_return_conditional_losses_481228o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������:���������:���������: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name481292:&"
 
_user_specified_name481290:&"
 
_user_specified_name481288:&"
 
_user_specified_name481286:&"
 
_user_specified_name481284:


_output_shapes
: :&	"
 
_user_specified_name481280:&"
 
_user_specified_name481278:

_output_shapes
: :&"
 
_user_specified_name481274:&"
 
_user_specified_name481272:

_output_shapes
: :&"
 
_user_specified_name481268:SO
#
_output_shapes
:���������
(
_user_specified_namePrice_Category:IE
#
_output_shapes
:���������

_user_specified_nameCity:M I
#
_output_shapes
:���������
"
_user_specified_name
Category
�
�
H__inference_sequential_7_layer_call_and_return_conditional_losses_480724
string_lookup_5_input>
:string_lookup_5_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_5_none_lookup_lookuptablefindv2_default_value	$
embedding_5_480720: 
identity��#embedding_5/StatefulPartitionedCall�-string_lookup_5/None_Lookup/LookupTableFindV2�
-string_lookup_5/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_5_none_lookup_lookuptablefindv2_table_handlestring_lookup_5_input;string_lookup_5_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
string_lookup_5/IdentityIdentity6string_lookup_5/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
#embedding_5/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_5/Identity:output:0embedding_5_480720*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_embedding_5_layer_call_and_return_conditional_losses_480708{
IdentityIdentity,embedding_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� x
NoOpNoOp$^embedding_5/StatefulPartitionedCall.^string_lookup_5/None_Lookup/LookupTableFindV2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: : : 2J
#embedding_5/StatefulPartitionedCall#embedding_5/StatefulPartitionedCall2^
-string_lookup_5/None_Lookup/LookupTableFindV2-string_lookup_5/None_Lookup/LookupTableFindV2:&"
 
_user_specified_name480720:

_output_shapes
: :,(
&
_user_specified_nametable_handle:Z V
#
_output_shapes
:���������
/
_user_specified_namestring_lookup_5_input
�
�
I__inference_sequential_10_layer_call_and_return_conditional_losses_481007
dense_2_input 
dense_2_481001:` 
dense_2_481003: 
identity��dense_2/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCalldense_2_inputdense_2_481001dense_2_481003*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_480991w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� D
NoOpNoOp ^dense_2/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������`: : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:&"
 
_user_specified_name481003:&"
 
_user_specified_name481001:V R
'
_output_shapes
:���������`
'
_user_specified_namedense_2_input
�
;
__inference__creator_481402
identity��
hash_tablen

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name101385*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: /
NoOpNoOp^hash_table*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table"�L
saver_filename:0StatefulPartitionedCall_4:0StatefulPartitionedCall_58"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
9
Category-
serving_default_Category:0���������
1
City)
serving_default_City:0���������
E
Price_Category3
 serving_default_Price_Category:0���������<
output_10
StatefulPartitionedCall:0���������
<
output_20
StatefulPartitionedCall:1���������
tensorflow/serving/predict:��
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
query_model
	identifiers
	_identifiers


candidates

_candidates
query_with_exclusions

signatures"
_tf_keras_model
Q
0
1
2
3
4
	5

6"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
trace_0
trace_12�
,__inference_brute_force_layer_call_fn_481263
,__inference_brute_force_layer_call_fn_481298�
���
FullArgSpec
args�
	jqueries
jk
varargs
 
varkw
 
defaults�

 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 ztrace_0ztrace_1
�
trace_0
trace_12�
G__inference_brute_force_layer_call_and_return_conditional_losses_481190
G__inference_brute_force_layer_call_and_return_conditional_losses_481228�
���
FullArgSpec
args�
	jqueries
jk
varargs
 
varkw
 
defaults�

 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 ztrace_0ztrace_1
�
	capture_1
	capture_4
	capture_7B�
!__inference__wrapped_model_480694CategoryCityPrice_Category"�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z	capture_1z	capture_4z	capture_7
�
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses
$user_embedding_model
%dense_layers"
_tf_keras_model
:�	2identifiers
:	�	 2
candidates
�2��
���
FullArgSpec)
args!�
	jqueries
j
exclusions
jk
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
annotations� *
 
,
&serving_default"
signature_map
(:& 2embedding_5/embeddings
(:& 2embedding_6/embeddings
(:& 2embedding_7/embeddings
 :` 2dense_2/kernel
: 2dense_2/bias
.
	0

1"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�
	capture_1
	capture_4
	capture_7B�
,__inference_brute_force_layer_call_fn_481263CategoryCityPrice_Category"�
���
FullArgSpec
args�
	jqueries
jk
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining
kwonlydefaults
 
annotations� *
 z	capture_1z	capture_4z	capture_7
�
	capture_1
	capture_4
	capture_7B�
,__inference_brute_force_layer_call_fn_481298CategoryCityPrice_Category"�
���
FullArgSpec
args�
	jqueries
jk
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining
kwonlydefaults
 
annotations� *
 z	capture_1z	capture_4z	capture_7
�
	capture_1
	capture_4
	capture_7B�
G__inference_brute_force_layer_call_and_return_conditional_losses_481190CategoryCityPrice_Category"�
���
FullArgSpec
args�
	jqueries
jk
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining
kwonlydefaults
 
annotations� *
 z	capture_1z	capture_4z	capture_7
�
	capture_1
	capture_4
	capture_7B�
G__inference_brute_force_layer_call_and_return_conditional_losses_481228CategoryCityPrice_Category"�
���
FullArgSpec
args�
	jqueries
jk
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining
kwonlydefaults
 
annotations� *
 z	capture_1z	capture_4z	capture_7
!J	
Const_8jtf.TrackableConstant
!J	
Const_7jtf.TrackableConstant
!J	
Const_6jtf.TrackableConstant
C
0
1
2
3
4"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
�
'non_trainable_variables

(layers
)metrics
*layer_regularization_losses
+layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
�
,trace_0
-trace_12�
.__inference_query_model_1_layer_call_fn_481123
.__inference_query_model_1_layer_call_fn_481152�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z,trace_0z-trace_1
�
.trace_0
/trace_12�
I__inference_query_model_1_layer_call_and_return_conditional_losses_481064
I__inference_query_model_1_layer_call_and_return_conditional_losses_481094�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z.trace_0z/trace_1
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses
6city_embedding
7category_embedding
8price_embedding"
_tf_keras_model
�
9layer_with_weights-0
9layer-0
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses"
_tf_keras_sequential
�
	capture_1
	capture_4
	capture_7B�
$__inference_signature_wrapper_481334CategoryCityPrice_Category"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 7

kwonlyargs)�&

jCategory
jCity
jPrice_Category
kwonlydefaults
 
annotations� *
 z	capture_1z	capture_4z	capture_7
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�
	capture_1
	capture_4
	capture_7B�
.__inference_query_model_1_layer_call_fn_481123CategoryCityPrice_Category"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining
kwonlydefaults
 
annotations� *
 z	capture_1z	capture_4z	capture_7
�
	capture_1
	capture_4
	capture_7B�
.__inference_query_model_1_layer_call_fn_481152CategoryCityPrice_Category"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining
kwonlydefaults
 
annotations� *
 z	capture_1z	capture_4z	capture_7
�
	capture_1
	capture_4
	capture_7B�
I__inference_query_model_1_layer_call_and_return_conditional_losses_481064CategoryCityPrice_Category"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining
kwonlydefaults
 
annotations� *
 z	capture_1z	capture_4z	capture_7
�
	capture_1
	capture_4
	capture_7B�
I__inference_query_model_1_layer_call_and_return_conditional_losses_481094CategoryCityPrice_Category"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining
kwonlydefaults
 
annotations� *
 z	capture_1z	capture_4z	capture_7
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
�
Etrace_0
Ftrace_12�
-__inference_user_model_1_layer_call_fn_480954
-__inference_user_model_1_layer_call_fn_480979�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 zEtrace_0zFtrace_1
�
Gtrace_0
Htrace_12�
H__inference_user_model_1_layer_call_and_return_conditional_losses_480900
H__inference_user_model_1_layer_call_and_return_conditional_losses_480929�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 zGtrace_0zHtrace_1
�
Ilayer-0
Jlayer_with_weights-0
Jlayer-1
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses"
_tf_keras_sequential
�
Qlayer-0
Rlayer_with_weights-0
Rlayer-1
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses"
_tf_keras_sequential
�
Ylayer-0
Zlayer_with_weights-0
Zlayer-1
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses"
_tf_keras_sequential
�
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
�
ltrace_0
mtrace_12�
.__inference_sequential_10_layer_call_fn_481016
.__inference_sequential_10_layer_call_fn_481025�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zltrace_0zmtrace_1
�
ntrace_0
otrace_12�
I__inference_sequential_10_layer_call_and_return_conditional_losses_480998
I__inference_sequential_10_layer_call_and_return_conditional_losses_481007�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zntrace_0zotrace_1
 "
trackable_list_wrapper
5
60
71
82"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�
	capture_1
	capture_4
	capture_7B�
-__inference_user_model_1_layer_call_fn_480954CategoryCityPrice_Category"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining
kwonlydefaults
 
annotations� *
 z	capture_1z	capture_4z	capture_7
�
	capture_1
	capture_4
	capture_7B�
-__inference_user_model_1_layer_call_fn_480979CategoryCityPrice_Category"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining
kwonlydefaults
 
annotations� *
 z	capture_1z	capture_4z	capture_7
�
	capture_1
	capture_4
	capture_7B�
H__inference_user_model_1_layer_call_and_return_conditional_losses_480900CategoryCityPrice_Category"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining
kwonlydefaults
 
annotations� *
 z	capture_1z	capture_4z	capture_7
�
	capture_1
	capture_4
	capture_7B�
H__inference_user_model_1_layer_call_and_return_conditional_losses_480929CategoryCityPrice_Category"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining
kwonlydefaults
 
annotations� *
 z	capture_1z	capture_4z	capture_7
:
p	keras_api
qlookup_table"
_tf_keras_layer
�
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
�
}trace_0
~trace_12�
-__inference_sequential_7_layer_call_fn_480735
-__inference_sequential_7_layer_call_fn_480746�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z}trace_0z~trace_1
�
trace_0
�trace_12�
H__inference_sequential_7_layer_call_and_return_conditional_losses_480713
H__inference_sequential_7_layer_call_and_return_conditional_losses_480724�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0z�trace_1
<
�	keras_api
�lookup_table"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
-__inference_sequential_8_layer_call_fn_480794
-__inference_sequential_8_layer_call_fn_480805�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
H__inference_sequential_8_layer_call_and_return_conditional_losses_480772
H__inference_sequential_8_layer_call_and_return_conditional_losses_480783�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
<
�	keras_api
�lookup_table"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
-__inference_sequential_9_layer_call_fn_480853
-__inference_sequential_9_layer_call_fn_480864�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
H__inference_sequential_9_layer_call_and_return_conditional_losses_480831
H__inference_sequential_9_layer_call_and_return_conditional_losses_480842�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_2_layer_call_fn_481343�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_2_layer_call_and_return_conditional_losses_481353�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
 "
trackable_list_wrapper
'
90"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_sequential_10_layer_call_fn_481016dense_2_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
.__inference_sequential_10_layer_call_fn_481025dense_2_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_sequential_10_layer_call_and_return_conditional_losses_480998dense_2_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_sequential_10_layer_call_and_return_conditional_losses_481007dense_2_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
_generic_user_object
j
�_initializer
�_create_resource
�_initialize
�_destroy_resourceR jtf.StaticHashTable
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_embedding_5_layer_call_fn_481360�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
G__inference_embedding_5_layer_call_and_return_conditional_losses_481368�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
 "
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�
	capture_1B�
-__inference_sequential_7_layer_call_fn_480735string_lookup_5_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z	capture_1
�
	capture_1B�
-__inference_sequential_7_layer_call_fn_480746string_lookup_5_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z	capture_1
�
	capture_1B�
H__inference_sequential_7_layer_call_and_return_conditional_losses_480713string_lookup_5_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z	capture_1
�
	capture_1B�
H__inference_sequential_7_layer_call_and_return_conditional_losses_480724string_lookup_5_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z	capture_1
"
_generic_user_object
j
�_initializer
�_create_resource
�_initialize
�_destroy_resourceR jtf.StaticHashTable
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_embedding_6_layer_call_fn_481375�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
G__inference_embedding_6_layer_call_and_return_conditional_losses_481383�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
 "
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�
	capture_1B�
-__inference_sequential_8_layer_call_fn_480794string_lookup_6_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z	capture_1
�
	capture_1B�
-__inference_sequential_8_layer_call_fn_480805string_lookup_6_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z	capture_1
�
	capture_1B�
H__inference_sequential_8_layer_call_and_return_conditional_losses_480772string_lookup_6_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z	capture_1
�
	capture_1B�
H__inference_sequential_8_layer_call_and_return_conditional_losses_480783string_lookup_6_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z	capture_1
"
_generic_user_object
j
�_initializer
�_create_resource
�_initialize
�_destroy_resourceR jtf.StaticHashTable
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_embedding_7_layer_call_fn_481390�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
G__inference_embedding_7_layer_call_and_return_conditional_losses_481398�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
 "
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�
	capture_1B�
-__inference_sequential_9_layer_call_fn_480853string_lookup_7_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z	capture_1
�
	capture_1B�
-__inference_sequential_9_layer_call_fn_480864string_lookup_7_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z	capture_1
�
	capture_1B�
H__inference_sequential_9_layer_call_and_return_conditional_losses_480831string_lookup_7_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z	capture_1
�
	capture_1B�
H__inference_sequential_9_layer_call_and_return_conditional_losses_480842string_lookup_7_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z	capture_1
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
�B�
(__inference_dense_2_layer_call_fn_481343inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
C__inference_dense_2_layer_call_and_return_conditional_losses_481353inputs"�
���
FullArgSpec
args�

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
annotations� *
 
"
_generic_user_object
�
�trace_02�
__inference__creator_481402�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference__initializer_481409�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference__destroyer_481413�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
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
�B�
,__inference_embedding_5_layer_call_fn_481360inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
G__inference_embedding_5_layer_call_and_return_conditional_losses_481368inputs"�
���
FullArgSpec
args�

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
annotations� *
 
"
_generic_user_object
�
�trace_02�
__inference__creator_481417�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference__initializer_481424�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference__destroyer_481428�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
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
�B�
,__inference_embedding_6_layer_call_fn_481375inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
G__inference_embedding_6_layer_call_and_return_conditional_losses_481383inputs"�
���
FullArgSpec
args�

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
annotations� *
 
"
_generic_user_object
�
�trace_02�
__inference__creator_481432�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference__initializer_481439�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference__destroyer_481443�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
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
�B�
,__inference_embedding_7_layer_call_fn_481390inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
G__inference_embedding_7_layer_call_and_return_conditional_losses_481398inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
__inference__creator_481402"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�
�	capture_1
�	capture_2B�
__inference__initializer_481409"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�	capture_1z�	capture_2
�B�
__inference__destroyer_481413"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference__creator_481417"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�
�	capture_1
�	capture_2B�
__inference__initializer_481424"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�	capture_1z�	capture_2
�B�
__inference__destroyer_481428"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference__creator_481432"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�
�	capture_1
�	capture_2B�
__inference__initializer_481439"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�	capture_1z�	capture_2
�B�
__inference__destroyer_481443"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
!J	
Const_5jtf.TrackableConstant
!J	
Const_4jtf.TrackableConstant
!J	
Const_3jtf.TrackableConstant
!J	
Const_2jtf.TrackableConstant
!J	
Const_1jtf.TrackableConstant
J
Constjtf.TrackableConstant@
__inference__creator_481402!�

� 
� "�
unknown @
__inference__creator_481417!�

� 
� "�
unknown @
__inference__creator_481432!�

� 
� "�
unknown B
__inference__destroyer_481413!�

� 
� "�
unknown B
__inference__destroyer_481428!�

� 
� "�
unknown B
__inference__destroyer_481443!�

� 
� "�
unknown K
__inference__initializer_481409(q���

� 
� "�
unknown L
__inference__initializer_481424)����

� 
� "�
unknown L
__inference__initializer_481439)����

� 
� "�
unknown �
!__inference__wrapped_model_480694�q��
	���
���
���
*
Category�
Category���������
"
City�
City���������
6
Price_Category$�!
Price_Category���������
� "c�`
.
output_1"�
output_1���������

.
output_2"�
output_2���������
�
G__inference_brute_force_layer_call_and_return_conditional_losses_481190�q��
	���
���
���
*
Category�
Category���������
"
City�
City���������
6
Price_Category$�!
Price_Category���������

 
�

trainingp"Y�V
O�L
$�!

tensor_0_0���������

$�!

tensor_0_1���������

� �
G__inference_brute_force_layer_call_and_return_conditional_losses_481228�q��
	���
���
���
*
Category�
Category���������
"
City�
City���������
6
Price_Category$�!
Price_Category���������

 
�

trainingp "Y�V
O�L
$�!

tensor_0_0���������

$�!

tensor_0_1���������

� �
,__inference_brute_force_layer_call_fn_481263�q��
	���
���
���
*
Category�
Category���������
"
City�
City���������
6
Price_Category$�!
Price_Category���������

 
�

trainingp"K�H
"�
tensor_0���������

"�
tensor_1���������
�
,__inference_brute_force_layer_call_fn_481298�q��
	���
���
���
*
Category�
Category���������
"
City�
City���������
6
Price_Category$�!
Price_Category���������

 
�

trainingp "K�H
"�
tensor_0���������

"�
tensor_1���������
�
C__inference_dense_2_layer_call_and_return_conditional_losses_481353c/�,
%�"
 �
inputs���������`
� ",�)
"�
tensor_0��������� 
� �
(__inference_dense_2_layer_call_fn_481343X/�,
%�"
 �
inputs���������`
� "!�
unknown��������� �
G__inference_embedding_5_layer_call_and_return_conditional_losses_481368^+�(
!�
�
inputs���������	
� ",�)
"�
tensor_0��������� 
� �
,__inference_embedding_5_layer_call_fn_481360S+�(
!�
�
inputs���������	
� "!�
unknown��������� �
G__inference_embedding_6_layer_call_and_return_conditional_losses_481383^+�(
!�
�
inputs���������	
� ",�)
"�
tensor_0��������� 
� �
,__inference_embedding_6_layer_call_fn_481375S+�(
!�
�
inputs���������	
� "!�
unknown��������� �
G__inference_embedding_7_layer_call_and_return_conditional_losses_481398^+�(
!�
�
inputs���������	
� ",�)
"�
tensor_0��������� 
� �
,__inference_embedding_7_layer_call_fn_481390S+�(
!�
�
inputs���������	
� "!�
unknown��������� �
I__inference_query_model_1_layer_call_and_return_conditional_losses_481064�q�����
���
���
*
Category�
Category���������
"
City�
City���������
6
Price_Category$�!
Price_Category���������
�

trainingp",�)
"�
tensor_0��������� 
� �
I__inference_query_model_1_layer_call_and_return_conditional_losses_481094�q�����
���
���
*
Category�
Category���������
"
City�
City���������
6
Price_Category$�!
Price_Category���������
�

trainingp ",�)
"�
tensor_0��������� 
� �
.__inference_query_model_1_layer_call_fn_481123�q�����
���
���
*
Category�
Category���������
"
City�
City���������
6
Price_Category$�!
Price_Category���������
�

trainingp"!�
unknown��������� �
.__inference_query_model_1_layer_call_fn_481152�q�����
���
���
*
Category�
Category���������
"
City�
City���������
6
Price_Category$�!
Price_Category���������
�

trainingp "!�
unknown��������� �
I__inference_sequential_10_layer_call_and_return_conditional_losses_480998r>�;
4�1
'�$
dense_2_input���������`
p

 
� ",�)
"�
tensor_0��������� 
� �
I__inference_sequential_10_layer_call_and_return_conditional_losses_481007r>�;
4�1
'�$
dense_2_input���������`
p 

 
� ",�)
"�
tensor_0��������� 
� �
.__inference_sequential_10_layer_call_fn_481016g>�;
4�1
'�$
dense_2_input���������`
p

 
� "!�
unknown��������� �
.__inference_sequential_10_layer_call_fn_481025g>�;
4�1
'�$
dense_2_input���������`
p 

 
� "!�
unknown��������� �
H__inference_sequential_7_layer_call_and_return_conditional_losses_480713wqB�?
8�5
+�(
string_lookup_5_input���������
p

 
� ",�)
"�
tensor_0��������� 
� �
H__inference_sequential_7_layer_call_and_return_conditional_losses_480724wqB�?
8�5
+�(
string_lookup_5_input���������
p 

 
� ",�)
"�
tensor_0��������� 
� �
-__inference_sequential_7_layer_call_fn_480735lqB�?
8�5
+�(
string_lookup_5_input���������
p

 
� "!�
unknown��������� �
-__inference_sequential_7_layer_call_fn_480746lqB�?
8�5
+�(
string_lookup_5_input���������
p 

 
� "!�
unknown��������� �
H__inference_sequential_8_layer_call_and_return_conditional_losses_480772x�B�?
8�5
+�(
string_lookup_6_input���������
p

 
� ",�)
"�
tensor_0��������� 
� �
H__inference_sequential_8_layer_call_and_return_conditional_losses_480783x�B�?
8�5
+�(
string_lookup_6_input���������
p 

 
� ",�)
"�
tensor_0��������� 
� �
-__inference_sequential_8_layer_call_fn_480794m�B�?
8�5
+�(
string_lookup_6_input���������
p

 
� "!�
unknown��������� �
-__inference_sequential_8_layer_call_fn_480805m�B�?
8�5
+�(
string_lookup_6_input���������
p 

 
� "!�
unknown��������� �
H__inference_sequential_9_layer_call_and_return_conditional_losses_480831x�B�?
8�5
+�(
string_lookup_7_input���������
p

 
� ",�)
"�
tensor_0��������� 
� �
H__inference_sequential_9_layer_call_and_return_conditional_losses_480842x�B�?
8�5
+�(
string_lookup_7_input���������
p 

 
� ",�)
"�
tensor_0��������� 
� �
-__inference_sequential_9_layer_call_fn_480853m�B�?
8�5
+�(
string_lookup_7_input���������
p

 
� "!�
unknown��������� �
-__inference_sequential_9_layer_call_fn_480864m�B�?
8�5
+�(
string_lookup_7_input���������
p 

 
� "!�
unknown��������� �
$__inference_signature_wrapper_481334�q��
	���
� 
���
*
Category�
category���������
"
City�
city���������
6
Price_Category$�!
price_category���������"c�`
.
output_1"�
output_1���������

.
output_2"�
output_2���������
�
H__inference_user_model_1_layer_call_and_return_conditional_losses_480900�q�����
���
���
*
Category�
Category���������
"
City�
City���������
6
Price_Category$�!
Price_Category���������
�

trainingp",�)
"�
tensor_0���������`
� �
H__inference_user_model_1_layer_call_and_return_conditional_losses_480929�q�����
���
���
*
Category�
Category���������
"
City�
City���������
6
Price_Category$�!
Price_Category���������
�

trainingp ",�)
"�
tensor_0���������`
� �
-__inference_user_model_1_layer_call_fn_480954�q�����
���
���
*
Category�
Category���������
"
City�
City���������
6
Price_Category$�!
Price_Category���������
�

trainingp"!�
unknown���������`�
-__inference_user_model_1_layer_call_fn_480979�q�����
���
���
*
Category�
Category���������
"
City�
City���������
6
Price_Category$�!
Price_Category���������
�

trainingp "!�
unknown���������`