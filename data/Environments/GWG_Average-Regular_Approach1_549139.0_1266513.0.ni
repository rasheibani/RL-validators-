[define indoor_room]
indoor_room is a kind of room.
indoor_room has a text called description.
indoor_room has a text called printed name.

[define area]
area is a kind of room.
area has a text called description.
area has a text called printed name.
area has a indoor_room called parent.
area has a text called X.
area has a text called Y.
area can be enterable.
area is always enterable.

player has a number called orientation.


[define landmark]
landmark is a kind of thing.
landmark has a text called description.
landmark has a text called printed name.
landmark can be examined.
area has a list of landmark called visible_objects.

[create r0]
r0 is a indoor_room. "Room 0".

[create r1]
r1 is a indoor_room. "Room 1".

[create r2]
r2 is a indoor_room. "Room 2".

[create r3]
r3 is a indoor_room. "Room 3".

[create r4]
r4 is a indoor_room. "Room 4".

[create r5]
r5 is a indoor_room. "Room 5".

[create a0r0]
a0r0 is a area. "An area (0) in r0".
the printed name of the a0r0 is "Room 0".
Understand "Area 0 in Room 0" as a0r0.
the parent of the a0r0 is r0.
the X of the a0r0 is "545737.0".
the Y of the a0r0 is "1265963.0".

[create a1r0]
a1r0 is a area. "An area (1) in r0".
the printed name of the a1r0 is "Room 0".
Understand "Area 1 in Room 0" as a1r0.
the parent of the a1r0 is r0.
the X of the a1r0 is "545773.0".
the Y of the a1r0 is "1266055.3867768592".

[create a6r0]
a6r0 is a area. "An area (6) in r0".
the printed name of the a6r0 is "Room 0".
Understand "Area 6 in Room 0" as a6r0.
the parent of the a6r0 is r0.
the X of the a6r0 is "547080.0".
the Y of the a6r0 is "1266169.0".

[create a7r0]
a7r0 is a area. "An area (7) in r0".
the printed name of the a7r0 is "Room 0".
Understand "Area 7 in Room 0" as a7r0.
the parent of the a7r0 is r0.
the X of the a7r0 is "547023.162688971".
the Y of the a7r0 is "1266206.4788710955".

[create a8r0]
a8r0 is a area. "An area (8) in r0".
the printed name of the a8r0 is "Room 0".
Understand "Area 8 in Room 0" as a8r0.
the parent of the a8r0 is r0.
the X of the a8r0 is "545147.0".
the Y of the a8r0 is "1266513.0".

[create a9r0]
a9r0 is a area. "An area (9) in r0".
the printed name of the a9r0 is "Room 0".
Understand "Area 9 in Room 0" as a9r0.
the parent of the a9r0 is r0.
the X of the a9r0 is "545231.2818622056".
the Y of the a9r0 is "1266549.642857143".

[create a14r0]
a14r0 is a area. "An area (14) in r0".
the printed name of the a14r0 is "Room 0".
Understand "Area 14 in Room 0" as a14r0.
the parent of the a14r0 is r0.
the X of the a14r0 is "546524.0".
the Y of the a14r0 is "1266854.0".

[create a15r0]
a15r0 is a area. "An area (15) in r0".
the printed name of the a15r0 is "Room 0".
Understand "Area 15 in Room 0" as a15r0.
the parent of the a15r0 is r0.
the X of the a15r0 is "546566.2240682532".
the Y of the a15r0 is "1266872.207678491".

[create a38r0]
a38r0 is a area. "An area (38) in r0".
the printed name of the a38r0 is "Room 0".
Understand "Area 38 in Room 0" as a38r0.
the parent of the a38r0 is r0.
the X of the a38r0 is "546533.0".
the Y of the a38r0 is "1266943.0".

[create a39r0]
a39r0 is a area. "An area (39) in r0".
the printed name of the a39r0 is "Room 0".
Understand "Area 39 in Room 0" as a39r0.
the parent of the a39r0 is r0.
the X of the a39r0 is "546564.0420577896".
the Y of the a39r0 is "1266917.3833874147".

[create a50r0]
a50r0 is a area. "An area (50) in r0".
the printed name of the a50r0 is "Room 0".
Understand "Area 50 in Room 0" as a50r0.
the parent of the a50r0 is r0.
the X of the a50r0 is "547561.0".
the Y of the a50r0 is "1266000.0".

[create a51r0]
a51r0 is a area. "An area (51) in r0".
the printed name of the a51r0 is "Room 0".
Understand "Area 51 in Room 0" as a51r0.
the parent of the a51r0 is r0.
the X of the a51r0 is "547520.2968536066".
the Y of the a51r0 is "1266051.4209743899".

[create a58r0]
a58r0 is a area. "An area (58) in r0".
the printed name of the a58r0 is "Room 0".
Understand "Area 58 in Room 0" as a58r0.
the parent of the a58r0 is r0.
the X of the a58r0 is "547279.0".
the Y of the a58r0 is "1265972.0".

[create a59r0]
a59r0 is a area. "An area (59) in r0".
the printed name of the a59r0 is "Room 0".
Understand "Area 59 in Room 0" as a59r0.
the parent of the a59r0 is r0.
the X of the a59r0 is "547310.1281488211".
the Y of the a59r0 is "1266017.0129297304".

[create a62r0]
a62r0 is a area. "An area (62) in r0".
the printed name of the a62r0 is "Room 0".
Understand "Area 62 in Room 0" as a62r0.
the parent of the a62r0 is r0.
the X of the a62r0 is "547412.791054277".
the Y of the a62r0 is "1266164.037205347".

[create a63r0]
a63r0 is a area. "An area (63) in r0".
the printed name of the a63r0 is "Room 0".
Understand "Area 63 in Room 0" as a63r0.
the parent of the a63r0 is r0.
the X of the a63r0 is "547426.7496886398".
the Y of the a63r0 is "1266162.317373606".

[create a65r0]
a65r0 is a area. "An area (65) in r0".
the printed name of the a65r0 is "Room 0".
Understand "Area 65 in Room 0" as a65r0.
the parent of the a65r0 is r0.
the X of the a65r0 is "546938.3936891638".
the Y of the a65r0 is "1266267.623157472".

[create a66r0]
a66r0 is a area. "An area (66) in r0".
the printed name of the a66r0 is "Room 0".
Understand "Area 66 in Room 0" as a66r0.
the parent of the a66r0 is r0.
the X of the a66r0 is "545327.6742984811".
the Y of the a66r0 is "1266591.579100882".

[create a69r0]
a69r0 is a area. "An area (69) in r0".
the printed name of the a69r0 is "Room 0".
Understand "Area 69 in Room 0" as a69r0.
the parent of the a69r0 is r0.
the X of the a69r0 is "546581.020066252".
the Y of the a69r0 is "1266889.2230761899".

[create a71r0]
a71r0 is a area. "An area (71) in r0".
the printed name of the a71r0 is "Room 0".
Understand "Area 71 in Room 0" as a71r0.
the parent of the a71r0 is r0.
the X of the a71r0 is "545790.3005528575".
the Y of the a71r0 is "1266098.8583478576".

[create a87r0]
a87r0 is a area. "An area (87) in r0".
the printed name of the a87r0 is "Room 0".
Understand "Area 87 in Room 0" as a87r0.
the parent of the a87r0 is r0.
the X of the a87r0 is "546949.2714746743".
the Y of the a87r0 is "1266757.2153626725".

[create a89r0]
a89r0 is a area. "An area (89) in r0".
the printed name of the a89r0 is "Room 0".
Understand "Area 89 in Room 0" as a89r0.
the parent of the a89r0 is r0.
the X of the a89r0 is "547165.8732540559".
the Y of the a89r0 is "1266825.1357839613".

[create a28r1]
a28r1 is a area. "An area (28) in r1".
the printed name of the a28r1 is "Room 1".
Understand "Area 28 in Room 1" as a28r1.
the parent of the a28r1 is r1.
the X of the a28r1 is "548167.0".
the Y of the a28r1 is "1268104.0".

[create a29r1]
a29r1 is a area. "An area (29) in r1".
the printed name of the a29r1 is "Room 1".
Understand "Area 29 in Room 1" as a29r1.
the parent of the a29r1 is r1.
the X of the a29r1 is "548133.6666666669".
the Y of the a29r1 is "1268053.536449501".

[create a56r1]
a56r1 is a area. "An area (56) in r1".
the printed name of the a56r1 is "Room 1".
Understand "Area 56 in Room 1" as a56r1.
the parent of the a56r1 is r1.
the X of the a56r1 is "547967.0".
the Y of the a56r1 is "1268104.0".

[create a57r1]
a57r1 is a area. "An area (57) in r1".
the printed name of the a57r1 is "Room 1".
Understand "Area 57 in Room 1" as a57r1.
the parent of the a57r1 is r1.
the X of the a57r1 is "548000.333333333".
the Y of the a57r1 is "1268060.9694444442".

[create a77r1]
a77r1 is a area. "An area (77) in r1".
the printed name of the a77r1 is "Room 1".
Understand "Area 77 in Room 1" as a77r1.
the parent of the a77r1 is r1.
the X of the a77r1 is "548067.5167685037".
the Y of the a77r1 is "1267973.513395759".

[create a86r1]
a86r1 is a area. "An area (86) in r1".
the printed name of the a86r1 is "Room 1".
Understand "Area 86 in Room 1" as a86r1.
the parent of the a86r1 is r1.
the X of the a86r1 is "548072.8708448356".
the Y of the a86r1 is "1267805.3162913683".

[create a48r2]
a48r2 is a area. "An area (48) in r2".
the printed name of the a48r2 is "Room 2".
Understand "Area 48 in Room 2" as a48r2.
the parent of the a48r2 is r2.
the X of the a48r2 is "548541.0".
the Y of the a48r2 is "1265972.0".

[create a49r2]
a49r2 is a area. "An area (49) in r2".
the printed name of the a49r2 is "Room 2".
Understand "Area 49 in Room 2" as a49r2.
the parent of the a49r2 is r2.
the X of the a49r2 is "548571.7043713275".
the Y of the a49r2 is "1266022.7067112252".

[create a52r2]
a52r2 is a area. "An area (52) in r2".
the printed name of the a52r2 is "Room 2".
Understand "Area 52 in Room 2" as a52r2.
the parent of the a52r2 is r2.
the X of the a52r2 is "548824.0".
the Y of the a52r2 is "1266000.0".

[create a53r2]
a53r2 is a area. "An area (53) in r2".
the printed name of the a53r2 is "Room 2".
Understand "Area 53 in Room 2" as a53r2.
the parent of the a53r2 is r2.
the X of the a53r2 is "548783.4539705468".
the Y of the a53r2 is "1266048.7643334025".

[create a60r2]
a60r2 is a area. "An area (60) in r2".
the printed name of the a60r2 is "Room 2".
Understand "Area 60 in Room 2" as a60r2.
the parent of the a60r2 is r2.
the X of the a60r2 is "548667.2390740267".
the Y of the a60r2 is "1266162.4312827692".

[create a61r2]
a61r2 is a area. "An area (61) in r2".
the printed name of the a61r2 is "Room 2".
Understand "Area 61 in Room 2" as a61r2.
the parent of the a61r2 is r2.
the X of the a61r2 is "548689.2896516463".
the Y of the a61r2 is "1266160.8833195835".

[create a16r3]
a16r3 is a area. "An area (16) in r3".
the printed name of the a16r3 is "Room 3".
Understand "Area 16 in Room 3" as a16r3.
the parent of the a16r3 is r3.
the X of the a16r3 is "549577.0".
the Y of the a16r3 is "1268114.0".

[create a17r3]
a17r3 is a area. "An area (17) in r3".
the printed name of the a17r3 is "Room 3".
Understand "Area 17 in Room 3" as a17r3.
the parent of the a17r3 is r3.
the X of the a17r3 is "549545.9123931624".
the Y of the a17r3 is "1268088.166239316".

[create a18r3]
a18r3 is a area. "An area (18) in r3".
the printed name of the a18r3 is "Room 3".
Understand "Area 18 in Room 3" as a18r3.
the parent of the a18r3 is r3.
the X of the a18r3 is "550949.0".
the Y of the a18r3 is "1267588.0".

[create a19r3]
a19r3 is a area. "An area (19) in r3".
the printed name of the a19r3 is "Room 3".
Understand "Area 19 in Room 3" as a19r3.
the parent of the a19r3 is r3.
the X of the a19r3 is "550957.3098646167".
the Y of the a19r3 is "1267627.908152513".

[create a20r3]
a20r3 is a area. "An area (20) in r3".
the printed name of the a20r3 is "Room 3".
Understand "Area 20 in Room 3" as a20r3.
the parent of the a20r3 is r3.
the X of the a20r3 is "551035.0".
the Y of the a20r3 is "1267625.0".

[create a21r3]
a21r3 is a area. "An area (21) in r3".
the printed name of the a21r3 is "Room 3".
Understand "Area 21 in Room 3" as a21r3.
the parent of the a21r3 is r3.
the X of the a21r3 is "550996.9614546176".
the Y of the a21r3 is "1267654.1909433214".

[create a22r3]
a22r3 is a area. "An area (22) in r3".
the printed name of the a22r3 is "Room 3".
Understand "Area 22 in Room 3" as a22r3.
the parent of the a22r3 is r3.
the X of the a22r3 is "548809.0".
the Y of the a22r3 is "1268024.0".

[create a23r3]
a23r3 is a area. "An area (23) in r3".
the printed name of the a23r3 is "Room 3".
Understand "Area 23 in Room 3" as a23r3.
the parent of the a23r3 is r3.
the X of the a23r3 is "548849.7622699386".
the Y of the a23r3 is "1268042.6487730062".

[create a24r3]
a24r3 is a area. "An area (24) in r3".
the printed name of the a24r3 is "Room 3".
Understand "Area 24 in Room 3" as a24r3.
the parent of the a24r3 is r3.
the X of the a24r3 is "549586.0".
the Y of the a24r3 is "1268024.0".

[create a25r3]
a25r3 is a area. "An area (25) in r3".
the printed name of the a25r3 is "Room 3".
Understand "Area 25 in Room 3" as a25r3.
the parent of the a25r3 is r3.
the X of the a25r3 is "549545.6625777469".
the Y of the a25r3 is "1268042.6912577746".

[create a32r3]
a32r3 is a area. "An area (32) in r3".
the printed name of the a32r3 is "Room 3".
Understand "Area 32 in Room 3" as a32r3.
the parent of the a32r3 is r3.
the X of the a32r3 is "548818.0".
the Y of the a32r3 is "1268114.0".

[create a33r3]
a33r3 is a area. "An area (33) in r3".
the printed name of the a33r3 is "Room 3".
Understand "Area 33 in Room 3" as a33r3.
the parent of the a33r3 is r3.
the X of the a33r3 is "548848.9153780071".
the Y of the a33r3 is "1268088.1834621993".

[create a46r3]
a46r3 is a area. "An area (46) in r3".
the printed name of the a46r3 is "Room 3".
Understand "Area 46 in Room 3" as a46r3.
the parent of the a46r3 is r3.
the X of the a46r3 is "549205.225952813".
the Y of the a46r3 is "1267455.556261343".

[create a47r3]
a47r3 is a area. "An area (47) in r3".
the printed name of the a47r3 is "Room 3".
Understand "Area 47 in Room 3" as a47r3.
the parent of the a47r3 is r3.
the X of the a47r3 is "549211.3121597095".
the Y of the a47r3 is "1267491.0618572293".

[create a54r3]
a54r3 is a area. "An area (54) in r3".
the printed name of the a54r3 is "Room 3".
Understand "Area 54 in Room 3" as a54r3.
the parent of the a54r3 is r3.
the X of the a54r3 is "550921.0".
the Y of the a54r3 is "1268086.0".

[create a55r3]
a55r3 is a area. "An area (55) in r3".
the printed name of the a55r3 is "Room 3".
Understand "Area 55 in Room 3" as a55r3.
the parent of the a55r3 is r3.
the X of the a55r3 is "550962.7727272727".
the Y of the a55r3 is "1268081.0".

[create a73r3]
a73r3 is a area. "An area (73) in r3".
the printed name of the a73r3 is "Room 3".
Understand "Area 73 in Room 3" as a73r3.
the parent of the a73r3 is r3.
the X of the a73r3 is "549529.5509290387".
the Y of the a73r3 is "1268061.0062088706".

[create a74r3]
a74r3 is a area. "An area (74) in r3".
the printed name of the a74r3 is "Room 3".
Understand "Area 74 in Room 3" as a74r3.
the parent of the a74r3 is r3.
the X of the a74r3 is "550981.1150421607".
the Y of the a74r3 is "1267659.7027389589".

[create a75r3]
a75r3 is a area. "An area (75) in r3".
the printed name of the a75r3 is "Room 3".
Understand "Area 75 in Room 3" as a75r3.
the parent of the a75r3 is r3.
the X of the a75r3 is "548865.4921500897".
the Y of the a75r3 is "1268060.7986347189".

[create a80r3]
a80r3 is a area. "An area (80) in r3".
the printed name of the a80r3 is "Room 3".
Understand "Area 80 in Room 3" as a80r3.
the parent of the a80r3 is r3.
the X of the a80r3 is "549115.4174139355".
the Y of the a80r3 is "1267336.0966946804".

[create a81r3]
a81r3 is a area. "An area (81) in r3".
the printed name of the a81r3 is "Room 3".
Understand "Area 81 in Room 3" as a81r3.
the parent of the a81r3 is r3.
the X of the a81r3 is "549283.1804825793".
the Y of the a81r3 is "1267294.491453657".

[create a82r3]
a82r3 is a area. "An area (82) in r3".
the printed name of the a82r3 is "Room 3".
Understand "Area 82 in Room 3" as a82r3.
the parent of the a82r3 is r3.
the X of the a82r3 is "549116.6987700826".
the Y of the a82r3 is "1267514.5259778567".

[create a84r3]
a84r3 is a area. "An area (84) in r3".
the printed name of the a84r3 is "Room 3".
Understand "Area 84 in Room 3" as a84r3.
the parent of the a84r3 is r3.
the X of the a84r3 is "550907.8718497385".
the Y of the a84r3 is "1267995.9455539705".

[create a85r3]
a85r3 is a area. "An area (85) in r3".
the printed name of the a85r3 is "Room 3".
Understand "Area 85 in Room 3" as a85r3.
the parent of the a85r3 is r3.
the X of the a85r3 is "549199.9486065779".
the Y of the a85r3 is "1267978.1064398694".

[create a90r3]
a90r3 is a area. "An area (90) in r3".
the printed name of the a90r3 is "Room 3".
Understand "Area 90 in Room 3" as a90r3.
the parent of the a90r3 is r3.
the X of the a90r3 is "549032.5496991468".
the Y of the a90r3 is "1267177.4635455795".

[create a91r3]
a91r3 is a area. "An area (91) in r3".
the printed name of the a91r3 is "Room 3".
Understand "Area 91 in Room 3" as a91r3.
the parent of the a91r3 is r3.
the X of the a91r3 is "549014.3751060646".
the Y of the a91r3 is "1267107.0663285346".

[create a92r3]
a92r3 is a area. "An area (92) in r3".
the printed name of the a92r3 is "Room 3".
Understand "Area 92 in Room 3" as a92r3.
the parent of the a92r3 is r3.
the X of the a92r3 is "549300.6894454771".
the Y of the a92r3 is "1267107.2580645164".

[create a2r4]
a2r4 is a area. "An area (2) in r4".
the printed name of the a2r4 is "Room 4".
Understand "Area 2 in Room 4" as a2r4.
the parent of the a2r4 is r4.
the X of the a2r4 is "549729.0".
the Y of the a2r4 is "1265963.0".

[create a3r4]
a3r4 is a area. "An area (3) in r4".
the printed name of the a3r4 is "Room 4".
Understand "Area 3 in Room 4" as a3r4.
the parent of the a3r4 is r4.
the X of the a3r4 is "549765.0".
the Y of the a3r4 is "1266055.3867768594".

[create a4r4]
a4r4 is a area. "An area (4) in r4".
the printed name of the a4r4 is "Room 4".
Understand "Area 4 in Room 4" as a4r4.
the parent of the a4r4 is r4.
the X of the a4r4 is "551072.0".
the Y of the a4r4 is "1266169.0".

[create a5r4]
a5r4 is a area. "An area (5) in r4".
the printed name of the a5r4 is "Room 4".
Understand "Area 5 in Room 4" as a5r4.
the parent of the a5r4 is r4.
the X of the a5r4 is "551013.3376066436".
the Y of the a5r4 is "1266210.784647849".

[create a10r4]
a10r4 is a area. "An area (10) in r4".
the printed name of the a10r4 is "Room 4".
Understand "Area 10 in Room 4" as a10r4.
the parent of the a10r4 is r4.
the X of the a10r4 is "549139.0".
the Y of the a10r4 is "1266513.0".

[create a11r4]
a11r4 is a area. "An area (11) in r4".
the printed name of the a11r4 is "Room 4".
Understand "Area 11 in Room 4" as a11r4.
the parent of the a11r4 is r4.
the X of the a11r4 is "549223.2818622056".
the Y of the a11r4 is "1266549.642857143".

[create a12r4]
a12r4 is a area. "An area (12) in r4".
the printed name of the a12r4 is "Room 4".
Understand "Area 12 in Room 4" as a12r4.
the parent of the a12r4 is r4.
the X of the a12r4 is "551102.0".
the Y of the a12r4 is "1266786.0".

[create a13r4]
a13r4 is a area. "An area (13) in r4".
the printed name of the a13r4 is "Room 4".
Understand "Area 13 in Room 4" as a13r4.
the parent of the a13r4 is r4.
the X of the a13r4 is "551039.166666667".
the Y of the a13r4 is "1266845.833333333".

[create a34r4]
a34r4 is a area. "An area (34) in r4".
the printed name of the a34r4 is "Room 4".
Understand "Area 34 in Room 4" as a34r4.
the parent of the a34r4 is r4.
the X of the a34r4 is "550516.0".
the Y of the a34r4 is "1266854.0".

[create a35r4]
a35r4 is a area. "An area (35) in r4".
the printed name of the a35r4 is "Room 4".
Understand "Area 35 in Room 4" as a35r4.
the parent of the a35r4 is r4.
the X of the a35r4 is "550558.2240682532".
the Y of the a35r4 is "1266872.2076784912".

[create a36r4]
a36r4 is a area. "An area (36) in r4".
the printed name of the a36r4 is "Room 4".
Understand "Area 36 in Room 4" as a36r4.
the parent of the a36r4 is r4.
the X of the a36r4 is "551262.0".
the Y of the a36r4 is "1266854.0".

[create a37r4]
a37r4 is a area. "An area (37) in r4".
the printed name of the a37r4 is "Room 4".
Understand "Area 37 in Room 4" as a37r4.
the parent of the a37r4 is r4.
the X of the a37r4 is "551234.2704191618".
the Y of the a37r4 is "1266873.6734131738".

[create a40r4]
a40r4 is a area. "An area (40) in r4".
the printed name of the a40r4 is "Room 4".
Understand "Area 40 in Room 4" as a40r4.
the parent of the a40r4 is r4.
the X of the a40r4 is "550525.0".
the Y of the a40r4 is "1266943.0".

[create a41r4]
a41r4 is a area. "An area (41) in r4".
the printed name of the a41r4 is "Room 4".
Understand "Area 41 in Room 4" as a41r4.
the parent of the a41r4 is r4.
the X of the a41r4 is "550556.0420577894".
the Y of the a41r4 is "1266917.3833874147".

[create a42r4]
a42r4 is a area. "An area (42) in r4".
the printed name of the a42r4 is "Room 4".
Understand "Area 42 in Room 4" as a42r4.
the parent of the a42r4 is r4.
the X of the a42r4 is "551253.0".
the Y of the a42r4 is "1266943.0".

[create a43r4]
a43r4 is a area. "An area (43) in r4".
the printed name of the a43r4 is "Room 4".
Understand "Area 43 in Room 4" as a43r4.
the parent of the a43r4 is r4.
the X of the a43r4 is "551221.7862838916".
the Y of the a43r4 is "1266917.366028708".

[create a64r4]
a64r4 is a area. "An area (64) in r4".
the printed name of the a64r4 is "Room 4".
Understand "Area 64 in Room 4" as a64r4.
the parent of the a64r4 is r4.
the X of the a64r4 is "550929.6979495725".
the Y of the a64r4 is "1266266.6546164062".

[create a67r4]
a67r4 is a area. "An area (67) in r4".
the printed name of the a67r4 is "Room 4".
Understand "Area 67 in Room 4" as a67r4.
the parent of the a67r4 is r4.
the X of the a67r4 is "549319.674298481".
the Y of the a67r4 is "1266591.579100882".

[create a68r4]
a68r4 is a area. "An area (68) in r4".
the printed name of the a68r4 is "Room 4".
Understand "Area 68 in Room 4" as a68r4.
the parent of the a68r4 is r4.
the X of the a68r4 is "551036.8477146518".
the Y of the a68r4 is "1266847.3338316958".

[create a70r4]
a70r4 is a area. "An area (70) in r4".
the printed name of the a70r4 is "Room 4".
Understand "Area 70 in Room 4" as a70r4.
the parent of the a70r4 is r4.
the X of the a70r4 is "550573.0200662517".
the Y of the a70r4 is "1266889.2230761896".

[create a72r4]
a72r4 is a area. "An area (72) in r4".
the printed name of the a72r4 is "Room 4".
Understand "Area 72 in Room 4" as a72r4.
the parent of the a72r4 is r4.
the X of the a72r4 is "549782.3005528575".
the Y of the a72r4 is "1266098.8583478576".

[create a78r4]
a78r4 is a area. "An area (78) in r4".
the printed name of the a78r4 is "Room 4".
Understand "Area 78 in Room 4" as a78r4.
the parent of the a78r4 is r4.
the X of the a78r4 is "551207.8618197282".
the Y of the a78r4 is "1266894.1585884353".

[create a88r4]
a88r4 is a area. "An area (88) in r4".
the printed name of the a88r4 is "Room 4".
Understand "Area 88 in Room 4" as a88r4.
the parent of the a88r4 is r4.
the X of the a88r4 is "550927.2063772489".
the Y of the a88r4 is "1266768.270160742".

[create a26r5]
a26r5 is a area. "An area (26) in r5".
the printed name of the a26r5 is "Room 5".
Understand "Area 26 in Room 5" as a26r5.
the parent of the a26r5 is r5.
the X of the a26r5 is "547359.0".
the Y of the a26r5 is "1268024.0".

[create a27r5]
a27r5 is a area. "An area (27) in r5".
the printed name of the a27r5 is "Room 5".
Understand "Area 27 in Room 5" as a27r5.
the parent of the a27r5 is r5.
the X of the a27r5 is "547318.6686930092".
the Y of the a27r5 is "1268042.2965214453".

[create a30r5]
a30r5 is a area. "An area (30) in r5".
the printed name of the a30r5 is "Room 5".
Understand "Area 30 in Room 5" as a30r5.
the parent of the a30r5 is r5.
the X of the a30r5 is "547349.0".
the Y of the a30r5 is "1268114.0".

[create a31r5]
a31r5 is a area. "An area (31) in r5".
the printed name of the a31r5 is "Room 5".
Understand "Area 31 in Room 5" as a31r5.
the parent of the a31r5 is r5.
the X of the a31r5 is "547318.1050372067".
the Y of the a31r5 is "1268087.7894485788".

[create a44r5]
a44r5 is a area. "An area (44) in r5".
the printed name of the a44r5 is "Room 5".
Understand "Area 44 in Room 5" as a44r5.
the parent of the a44r5 is r5.
the X of the a44r5 is "545147.0".
the Y of the a44r5 is "1267327.0".

[create a45r5]
a45r5 is a area. "An area (45) in r5".
the printed name of the a45r5 is "Room 5".
Understand "Area 45 in Room 5" as a45r5.
the parent of the a45r5 is r5.
the X of the a45r5 is "545296.0294117647".
the Y of the a45r5 is "1267289.375".

[create a76r5]
a76r5 is a area. "An area (76) in r5".
the printed name of the a76r5 is "Room 5".
Understand "Area 76 in Room 5" as a76r5.
the parent of the a76r5 is r5.
the X of the a76r5 is "547302.1653756484".
the Y of the a76r5 is "1268061.2658517454".

[create a79r5]
a79r5 is a area. "An area (79) in r5".
the printed name of the a79r5 is "Room 5".
Understand "Area 79 in Room 5" as a79r5.
the parent of the a79r5 is r5.
the X of the a79r5 is "545306.903207838".
the Y of the a79r5 is "1267286.8473628089".

[create a83r5]
a83r5 is a area. "An area (83) in r5".
the printed name of the a83r5 is "Room 5".
Understand "Area 83 in Room 5" as a83r5.
the parent of the a83r5 is r5.
the X of the a83r5 is "546928.8152539522".
the Y of the a83r5 is "1267944.43178187".

[create d0]
d0 is a door. "A door between a61r2 and a91r3".
d0 is north of a61r2 and south of a91r3.

[create d2]
d2 is a door. "A door between a63r0 and a86r1".
d2 is north of a63r0 and south of a86r1.

[create d4]
d4 is a door. "A door between a86r1 and a60r2".
d4 is southeast of a86r1 and northwest of a60r2.

[create d6]
d6 is a door. "A door between a89r0 and a83r5".
d6 is north of a89r0 and south of a83r5.

[create d7]
d7 is a door. "A door between a66r0 and a79r5".
d7 is north of a66r0 and south of a79r5.

[create d8]
d8 is a door. "A door between a92r3 and a67r4".
d8 is south of a92r3 and north of a67r4.

north of a0r0 is south of a1r0.

north of a2r4 is south of a3r4.

northwest of a4r4 is southeast of a5r4.

northwest of a6r0 is southeast of a7r0.

northeast of a8r0 is southwest of a9r0.

northeast of a10r4 is southwest of a11r4.

northwest of a12r4 is southeast of a13r4.

northeast of a14r0 is southwest of a15r0.

southwest of a16r3 is northeast of a17r3.

north of a18r3 is south of a19r3.

northwest of a20r3 is southeast of a21r3.

northeast of a22r3 is southwest of a23r3.

northwest of a24r3 is southeast of a25r3.

northwest of a26r5 is southeast of a27r5.

southwest of a28r1 is northeast of a29r1.

southwest of a30r5 is northeast of a31r5.

southeast of a32r3 is northwest of a33r3.

northeast of a34r4 is southwest of a35r4.

northwest of a36r4 is southeast of a37r4.

southeast of a38r0 is northwest of a39r0.

southeast of a40r4 is northwest of a41r4.

southwest of a42r4 is northeast of a43r4.

east of a44r5 is west of a45r5.

north of a46r3 is south of a47r3.

northeast of a48r2 is southwest of a49r2.

northwest of a50r0 is southeast of a51r0.

northwest of a52r2 is southeast of a53r2.

east of a54r3 is west of a55r3.

southeast of a56r1 is northwest of a57r1.

northeast of a58r0 is southwest of a59r0.

east of a60r2 is west of a61r2.

east of a62r0 is west of a63r0.

northwest of a5r4 is southeast of a64r4.

northwest of a7r0 is southeast of a65r0.

northeast of a9r0 is southwest of a66r0.

northeast of a11r4 is southwest of a67r4.

northwest of a13r4 is southeast of a68r4.

northeast of a15r0 is southwest of a69r0.

northeast of a35r4 is southwest of a70r4.

north of a1r0 is south of a71r0.

north of a3r4 is south of a72r4.

southwest of a17r3 is northeast of a73r3.

northeast of a19r3 is southwest of a74r3.

west of a21r3 is east of a74r3.

northeast of a23r3 is southwest of a75r3.

northwest of a25r3 is southeast of a73r3.

northwest of a27r5 is southeast of a76r5.

southwest of a29r1 is northeast of a77r1.

southwest of a31r5 is northeast of a76r5.

southeast of a33r3 is northwest of a75r3.

northwest of a37r4 is southeast of a78r4.

southeast of a39r0 is northwest of a69r0.

southeast of a41r4 is northwest of a70r4.

southwest of a43r4 is northeast of a78r4.

east of a45r5 is west of a79r5.

east of a80r3 is west of a81r3.

west of a47r3 is east of a82r3.

west of a65r0 is east of a71r0.

west of a64r4 is east of a72r4.

east of a79r5 is west of a83r5.

northeast of a81r3 is southwest of a84r3.

north of a80r3 is south of a82r3.

north of a82r3 is south of a85r3.

north of a74r3 is south of a84r3.

south of a77r1 is north of a86r1.

southwest of a55r3 is northeast of a84r3.

southeast of a57r1 is northwest of a77r1.

west of a76r5 is east of a83r5.

east of a69r0 is west of a87r0.

southwest of a68r4 is northeast of a88r4.

east of a87r0 is west of a89r0.

east of a68r4 is west of a78r4.

south of a90r3 is north of a91r3.

south of a81r3 is north of a92r3.

southwest of a80r3 is northeast of a90r3.

north of a62r0 is south of a89r0.

north of a65r0 is south of a87r0.

north of a64r4 is south of a88r4.

east of a70r4 is west of a88r4.

northeast of a59r0 is southwest of a62r0.

northeast of a49r2 is southwest of a60r2.

southeast of a66r0 is northwest of a71r0.

southeast of a67r4 is northwest of a72r4.

northwest of a51r0 is southeast of a63r0.

northwest of a53r2 is southeast of a61r2.

east of a75r3 is west of a85r3.

west of a73r3 is east of a85r3.

Definition: a direction (called thatway) is viable if the room thatway from the location is not nowhere.

nLooking is a number that varies. nLooking is 0. 
dirNumber is a number that varies.
dirNumber is 0. 
relDirDesc is a text that varies. 

[describe areas and rooms]
After looking:
	now nLooking is 1; 
	let accessibleRooms be a list of rooms;
	let accessibleAreas be a list of areas;
	let pDirections be list of viable directions;
	let parentSource be the parent of the location of player;
	let relDirs be a list of number;
	repeat with dirToLookAt running through pDirections:
		try silently going dirToLookAt;
		if rule succeeded:
			now dirNumber is 0;
			if "[dirToLookAt]" is "south":
				now dirNumber is 4;
			if "[dirToLookAt]" is "east":
				now dirNumber is 6;
			if "[dirToLookAt]" is "west":
				now dirNumber is 2;
			if "[dirToLookAt]" is "northwest":
				now dirNumber is 1;
			if "[dirToLookAt]" is "southwest":
				now dirNumber is 3;
			if "[dirToLookAt]" is "southeast":
				now dirNumber is 5;
			if "[dirToLookAt]" is "northeast":
				now dirNumber is 7;
			let relDir be the remainder after dividing the orientation of the player - dirNumber + 80 by 8;
			add relDir to relDirs;
			if relDir is 0:
				now relDirDesc is "at the front";
			if relDir is 1:
				now relDirDesc is "at the slight right";
			if relDir is 2:
				now relDirDesc is "at the right";
			if relDir is 3:
				now relDirDesc is "at the sharp right";
			if relDir is 4:
				now relDirDesc is "at the back";
			if relDir is 5:
				now relDirDesc is "at the sharp left";
			if relDir is 6:
				now relDirDesc is "at the left";
			if relDir is 7:
				now relDirDesc is "at the slight left";
			let destinationParent be the parent of the location of the player;
			if "[parentSource]" is "[destinationParent]":
				say "You can continue in the [parentSource] by going [dirToLookAt] ([relDirDesc])[line break]";
				add the location of the player to accessibleAreas;
			otherwise:
				say "You can enter the [destinationParent] by going [dirToLookAt] ([relDirDesc])[line break]";
				add the destinationParent to accessibleRooms;
			try silently going the opposite of dirToLookAt;
	repeat with vO running through the visible_objects of the location:
		say "[vO] is visible from here, but too far! You can move in this room to examine or access [it]";
	now nLooking is 0.
Before going through a locked door when nLooking is 1:
	stop the action.
Going front is an action applying to nothing. Understand "go front" as going front.

Check going front:
	if the orientation of the player is:
		-- 0: try the player going north;
		-- 1: try the player going northwest;
		-- 2: try the player going west;
		-- 3: try the player going southwest;
		-- 4: try the player going south;
		-- 5: try the player going southeast;
		-- 6: try the player going east;
		-- 7: try the player going northeast;
		-- otherwise: say "Yaaaaaa Babaaaaam!!!".


Going back is an action applying to nothing. Understand "go back" as going back.

Check going back:
	if the orientation of the player is:
		-- 4: try the player going north;
		-- 5: try the player going northwest;
		-- 6: try the player going west;
		-- 7: try the player going southwest;
		-- 0: try the player going south;
		-- 1: try the player going southeast;
		-- 2: try the player going east;
		-- 3: try the player going northeast;
		-- otherwise: say "Yaaaaaa Babaaaaam!!!".



Going left is an action applying to nothing. Understand "go left" as going left.

Check going left:
	if the orientation of the player is:
		-- 6: try the player going north;
		-- 7: try the player going northwest;
		-- 0: try the player going west;
		-- 1: try the player going southwest;
		-- 2: try the player going south;
		-- 3: try the player going southeast;
		-- 4: try the player going east;
		-- 5: try the player going northeast;
		-- otherwise: say "Yaaaaaa Babaaaaam!!!".


Going right is an action applying to nothing. Understand "go right" as going right.

Check going right:
	if the orientation of the player is:
		-- 2: try the player going north;
		-- 3: try the player going northwest;
		-- 4: try the player going west;
		-- 5: try the player going southwest;
		-- 6: try the player going south;
		-- 7: try the player going southeast;
		-- 0: try the player going east;
		-- 1: try the player going northeast;
		-- otherwise: say "Yaaaaaa Babaaaaam!!!".

Going sharp right is an action applying to nothing. Understand "go sharp right" as going sharp right.
Going slight right is an action applying to nothing. Understand "go slight right" as going slight right.
Going sharp left is an action applying to nothing. Understand "go sharp left" as going sharp left.
Going slight left is an action applying to nothing. Understand "go slight left" as going slight left.

Check going sharp right:
	if the orientation of the player is:
		-- 3: try the player going north;
		-- 4: try the player going northwest;
		-- 5: try the player going west;
		-- 6: try the player going southwest;
		-- 7: try the player going south;
		-- 0: try the player going southeast;
		-- 1: try the player going east;
		-- 2: try the player going northeast;
		-- otherwise: say "Yaaaaaa Babaaaaam!!!".

Check going slight right:
	if the orientation of the player is:
		-- 1: try the player going north;
		-- 2: try the player going northwest;
		-- 3: try the player going west;
		-- 4: try the player going southwest;
		-- 5: try the player going south;
		-- 6: try the player going southeast;
		-- 7: try the player going east;
		-- 0: try the player going northeast;
		-- otherwise: say "Yaaaaaa Babaaaaam!!!".

Check going sharp left:
	if the orientation of the player is:
		-- 5: try the player going north;
		-- 6: try the player going northwest;
		-- 7: try the player going west;
		-- 0: try the player going southwest;
		-- 1: try the player going south;
		-- 2: try the player going southeast;
		-- 3: try the player going east;
		-- 4: try the player going northeast;
		-- otherwise: say "Yaaaaaa Babaaaaam!!!".

Check going slight left:
	if the orientation of the player is:
		-- 7: try the player going north;
		-- 0: try the player going northwest;
		-- 1: try the player going west;
		-- 2: try the player going southwest;
		-- 3: try the player going south;
		-- 4: try the player going southeast;
		-- 5: try the player going east;
		-- 6: try the player going northeast;
		-- otherwise: say "Yaaaaaa Babaaaaam!!!".


Instead of going north:
	if nLooking is 0:
		now the orientation of the player is 0;
	continue the action.

Instead of going south:
	if nLooking is 0:
		now the orientation of the player is 4;
	continue the action.

Instead of going east:
	if nLooking is 0:
		now the orientation of the player is 6;
	continue the action.

Instead of going west:
	if nLooking is 0:
		now the orientation of the player is 2;
	continue the action.

Instead of going northwest:
	if nLooking is 0:
		now the orientation of the player is 1;
	continue the action.

Instead of going southwest:
	if nLooking is 0:
		now the orientation of the player is 3;
	continue the action.

Instead of going northeast:
	if nLooking is 0:
		now the orientation of the player is 7;
	continue the action.

Instead of going southeast:
	if nLooking is 0:
		now the orientation of the player is 5;
	continue the action.

Understand "veer left" as going slight left.
Understand "veer right" as going slight right.
Understand "turn left" as going left.
Understand "turn right" as going right.
Understand "turn around" as going back.
Understand "turn sharp left" as going sharp left.
Understand "Go straight" as going front.
Understand "turn sharp right" as going sharp right.
Understand "Turn slightly left" as going slight left.
Understand "Turn slightly right" as going slight right.
Understand "Make a sharp right" as going sharp right.
Understand "Make a sharp left" as going sharp left.
Every turn:
	let xx be the X of the location of the player;
	let yy be the Y of the location of the player;
	say "X: [xx][line break]";
	say "Y: [yy]".
the player is in a10r4.

the orientation of the player is 6.
