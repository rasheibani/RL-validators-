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

[create r6]
r6 is a indoor_room. "Room 6".

[create r7]
r7 is a indoor_room. "Room 7".

[create a13r0]
a13r0 is a area. "An area (13) in r0".
the printed name of the a13r0 is "Room 0".
Understand "Area 13 in Room 0" as a13r0.
the parent of the a13r0 is r0.
the X of the a13r0 is "545903.7496886395".
the Y of the a13r0 is "1676562.3173736061".

[create a16r0]
a16r0 is a area. "An area (16) in r0".
the printed name of the a16r0 is "Room 0".
Understand "Area 16 in Room 0" as a16r0.
the parent of the a16r0 is r0.
the X of the a16r0 is "545890.0086022716".
the Y of the a16r0 is "1676564.0104014394".

[create a26r0]
a26r0 is a area. "An area (26) in r0".
the printed name of the a26r0 is "Room 0".
Understand "Area 26 in Room 0" as a26r0.
the parent of the a26r0 is r0.
the X of the a26r0 is "545786.8816727516".
the Y of the a26r0 is "1676419.4952958585".

[create a28r0]
a28r0 is a area. "An area (28) in r0".
the printed name of the a28r0 is "Room 0".
Understand "Area 28 in Room 0" as a28r0.
the parent of the a28r0 is r0.
the X of the a28r0 is "545997.2968536066".
the Y of the a28r0 is "1676451.42097439".

[create a86r0]
a86r0 is a area. "An area (86) in r0".
the printed name of the a86r0 is "Room 0".
Understand "Area 86 in Room 0" as a86r0.
the parent of the a86r0 is r0.
the X of the a86r0 is "546038.0".
the Y of the a86r0 is "1676400.0".

[create a92r0]
a92r0 is a area. "An area (92) in r0".
the printed name of the a92r0 is "Room 0".
Understand "Area 92 in Room 0" as a92r0.
the parent of the a92r0 is r0.
the X of the a92r0 is "545756.0".
the Y of the a92r0 is "1676372.0".

[create a14r1]
a14r1 is a area. "An area (14) in r1".
the printed name of the a14r1 is "Room 1".
Understand "Area 14 in Room 1" as a14r1.
the parent of the a14r1 is r1.
the X of the a14r1 is "546549.8708448356".
the Y of the a14r1 is "1678205.3162913683".

[create a30r1]
a30r1 is a area. "An area (30) in r1".
the printed name of the a30r1 is "Room 1".
Understand "Area 30 in Room 1" as a30r1.
the parent of the a30r1 is r1.
the X of the a30r1 is "546544.5167685037".
the Y of the a30r1 is "1678373.513395759".

[create a35r1]
a35r1 is a area. "An area (35) in r1".
the printed name of the a35r1 is "Room 1".
Understand "Area 35 in Room 1" as a35r1.
the parent of the a35r1 is r1.
the X of the a35r1 is "546477.3333333333".
the Y of the a35r1 is "1678460.9694444444".

[create a47r1]
a47r1 is a area. "An area (47) in r1".
the printed name of the a47r1 is "Room 1".
Understand "Area 47 in Room 1" as a47r1.
the parent of the a47r1 is r1.
the X of the a47r1 is "546644.0".
the Y of the a47r1 is "1678504.0".

[create a48r1]
a48r1 is a area. "An area (48) in r1".
the printed name of the a48r1 is "Room 1".
Understand "Area 48 in Room 1" as a48r1.
the parent of the a48r1 is r1.
the X of the a48r1 is "546610.6666666667".
the Y of the a48r1 is "1678453.5364495006".

[create a91r1]
a91r1 is a area. "An area (91) in r1".
the printed name of the a91r1 is "Room 1".
Understand "Area 91 in Room 1" as a91r1.
the parent of the a91r1 is r1.
the X of the a91r1 is "546444.0".
the Y of the a91r1 is "1678504.0".

[create a10r2]
a10r2 is a area. "An area (10) in r2".
the printed name of the a10r2 is "Room 2".
Understand "Area 10 in Room 2" as a10r2.
the parent of the a10r2 is r2.
the X of the a10r2 is "547166.2896516463".
the Y of the a10r2 is "1676560.883319583".

[create a11r2]
a11r2 is a area. "An area (11) in r2".
the printed name of the a11r2 is "Room 2".
Understand "Area 11 in Room 2" as a11r2.
the parent of the a11r2 is r2.
the X of the a11r2 is "547260.4539705468".
the Y of the a11r2 is "1676448.7643334023".

[create a17r2]
a17r2 is a area. "An area (17) in r2".
the printed name of the a17r2 is "Room 2".
Understand "Area 17 in Room 2" as a17r2.
the parent of the a17r2 is r2.
the X of the a17r2 is "547144.2390740269".
the Y of the a17r2 is "1676562.431282769".

[create a27r2]
a27r2 is a area. "An area (27) in r2".
the printed name of the a27r2 is "Room 2".
Understand "Area 27 in Room 2" as a27r2.
the parent of the a27r2 is r2.
the X of the a27r2 is "547048.7043713275".
the Y of the a27r2 is "1676422.706711225".

[create a85r2]
a85r2 is a area. "An area (85) in r2".
the printed name of the a85r2 is "Room 2".
Understand "Area 85 in Room 2" as a85r2.
the parent of the a85r2 is r2.
the X of the a85r2 is "547018.0".
the Y of the a85r2 is "1676372.0".

[create a87r2]
a87r2 is a area. "An area (87) in r2".
the printed name of the a87r2 is "Room 2".
Understand "Area 87 in Room 2" as a87r2.
the parent of the a87r2 is r2.
the X of the a87r2 is "547301.0".
the Y of the a87r2 is "1676400.0".

[create a0r3]
a0r3 is a area. "An area (0) in r3".
the printed name of the a0r3 is "Room 3".
Understand "Area 0 in Room 3" as a0r3.
the parent of the a0r3 is r3.
the X of the a0r3 is "548877.1509433963".
the Y of the a0r3 is "1677466.0".

[create a2r3]
a2r3 is a area. "An area (2) in r3".
the printed name of the a2r3 is "Room 3".
Understand "Area 2 in Room 3" as a2r3.
the parent of the a2r3 is r3.
the X of the a2r3 is "547999.0929783394".
the Y of the a2r3 is "1677466.0".

[create a3r3]
a3r3 is a area. "An area (3) in r3".
the printed name of the a3r3 is "Room 3".
Understand "Area 3 in Room 3" as a3r3.
the parent of the a3r3 is r3.
the X of the a3r3 is "547994.2203248538".
the Y of the a3r3 is "1678341.680266012".

[create a4r3]
a4r3 is a area. "An area (4) in r3".
the printed name of the a4r3 is "Room 3".
Understand "Area 4 in Room 3" as a4r3.
the parent of the a4r3 is r3.
the X of the a4r3 is "548919.4606160175".
the Y of the a4r3 is "1677722.258040583".

[create a5r3]
a5r3 is a area. "An area (5) in r3".
the printed name of the a5r3 is "Room 3".
Understand "Area 5 in Room 3" as a5r3.
the parent of the a5r3 is r3.
the X of the a5r3 is "548941.3859923373".
the Y of the a5r3 is "1678078.694461914".

[create a6r3]
a6r3 is a area. "An area (6) in r3".
the printed name of the a6r3 is "Room 3".
Understand "Area 6 in Room 3" as a6r3.
the parent of the a6r3 is r3.
the X of the a6r3 is "548939.1312549781".
the Y of the a6r3 is "1678341.8497264257".

[create a12r3]
a12r3 is a area. "An area (12) in r3".
the printed name of the a12r3 is "Room 3".
Understand "Area 12 in Room 3" as a12r3.
the parent of the a12r3 is r3.
the X of the a12r3 is "547677.1587808633".
the Y of the a12r3 is "1678381.3772771838".

[create a29r3]
a29r3 is a area. "An area (29) in r3".
the printed name of the a29r3 is "Room 3".
Understand "Area 29 in Room 3" as a29r3.
the parent of the a29r3 is r3.
the X of the a29r3 is "549005.6887968997".
the Y of the a29r3 is "1678397.0123126113".

[create a31r3]
a31r3 is a area. "An area (31) in r3".
the printed name of the a31r3 is "Room 3".
Understand "Area 31 in Room 3" as a31r3.
the parent of the a31r3 is r3.
the X of the a31r3 is "550070.0337224285".
the Y of the a31r3 is "1678383.5967467793".

[create a32r3]
a32r3 is a area. "An area (32) in r3".
the printed name of the a32r3 is "Room 3".
Understand "Area 32 in Room 3" as a32r3.
the parent of the a32r3 is r3.
the X of the a32r3 is "548569.0239959399".
the Y of the a32r3 is "1678445.031506458".

[create a33r3]
a33r3 is a area. "An area (33) in r3".
the printed name of the a33r3 is "Room 3".
Understand "Area 33 in Room 3" as a33r3.
the parent of the a33r3 is r3.
the X of the a33r3 is "547342.4921500896".
the Y of the a33r3 is "1678460.7986347186".

[create a34r3]
a34r3 is a area. "An area (34) in r3".
the printed name of the a34r3 is "Room 3".
Understand "Area 34 in Room 3" as a34r3.
the parent of the a34r3 is r3.
the X of the a34r3 is "547630.4944995975".
the Y of the a34r3 is "1678411.0205706109".

[create a38r3]
a38r3 is a area. "An area (38) in r3".
the printed name of the a38r3 is "Room 3".
Understand "Area 38 in Room 3" as a38r3.
the parent of the a38r3 is r3.
the X of the a38r3 is "548006.5675408728".
the Y of the a38r3 is "1678353.681944269".

[create a41r3]
a41r3 is a area. "An area (41) in r3".
the printed name of the a41r3 is "Room 3".
Understand "Area 41 in Room 3" as a41r3.
the parent of the a41r3 is r3.
the X of the a41r3 is "547286.0".
the Y of the a41r3 is "1678424.0".

[create a42r3]
a42r3 is a area. "An area (42) in r3".
the printed name of the a42r3 is "Room 3".
Understand "Area 42 in Room 3" as a42r3.
the parent of the a42r3 is r3.
the X of the a42r3 is "547326.7622699386".
the Y of the a42r3 is "1678442.6487730062".

[create a51r3]
a51r3 is a area. "An area (51) in r3".
the printed name of the a51r3 is "Room 3".
Understand "Area 51 in Room 3" as a51r3.
the parent of the a51r3 is r3.
the X of the a51r3 is "547295.0".
the Y of the a51r3 is "1678514.0".

[create a52r3]
a52r3 is a area. "An area (52) in r3".
the printed name of the a52r3 is "Room 3".
Understand "Area 52 in Room 3" as a52r3.
the parent of the a52r3 is r3.
the X of the a52r3 is "547325.915378007".
the Y of the a52r3 is "1678488.1834621993".

[create a55r3]
a55r3 is a area. "An area (55) in r3".
the printed name of the a55r3 is "Room 3".
Understand "Area 55 in Room 3" as a55r3.
the parent of the a55r3 is r3.
the X of the a55r3 is "548530.0".
the Y of the a55r3 is "1678504.0".

[create a56r3]
a56r3 is a area. "An area (56) in r3".
the printed name of the a56r3 is "Room 3".
Understand "Area 56 in Room 3" as a56r3.
the parent of the a56r3 is r3.
the X of the a56r3 is "548562.8264604809".
the Y of the a56r3 is "1678505.7673539517".

[create a69r3]
a69r3 is a area. "An area (69) in r3".
the printed name of the a69r3 is "Room 3".
Understand "Area 69 in Room 3" as a69r3.
the parent of the a69r3 is r3.
the X of the a69r3 is "548781.0".
the Y of the a69r3 is "1677822.0".

[create a70r3]
a70r3 is a area. "An area (70) in r3".
the printed name of the a70r3 is "Room 3".
Understand "Area 70 in Room 3" as a70r3.
the parent of the a70r3 is r3.
the X of the a70r3 is "548786.4038159978".
the Y of the a70r3 is "1677793.4390882854".

[create a71r3]
a71r3 is a area. "An area (71) in r3".
the printed name of the a71r3 is "Room 3".
Understand "Area 71 in Room 3" as a71r3.
the parent of the a71r3 is r3.
the X of the a71r3 is "549100.0".
the Y of the a71r3 is "1678025.0".

[create a72r3]
a72r3 is a area. "An area (72) in r3".
the printed name of the a72r3 is "Room 3".
Understand "Area 72 in Room 3" as a72r3.
the parent of the a72r3 is r3.
the X of the a72r3 is "549073.9056948908".
the Y of the a72r3 is "1678049.9230439267".

[create a79r3]
a79r3 is a area. "An area (79) in r3".
the printed name of the a79r3 is "Room 3".
Understand "Area 79 in Room 3" as a79r3.
the parent of the a79r3 is r3.
the X of the a79r3 is "547589.0".
the Y of the a79r3 is "1678514.0".

[create a80r3]
a80r3 is a area. "An area (80) in r3".
the printed name of the a80r3 is "Room 3".
Understand "Area 80 in Room 3" as a80r3.
the parent of the a80r3 is r3.
the X of the a80r3 is "547620.826460481".
the Y of the a80r3 is "1678505.767353952".

[create a81r3]
a81r3 is a area. "An area (81) in r3".
the printed name of the a81r3 is "Room 3".
Understand "Area 81 in Room 3" as a81r3.
the parent of the a81r3 is r3.
the X of the a81r3 is "548054.0".
the Y of the a81r3 is "1678514.0".

[create a82r3]
a82r3 is a area. "An area (82) in r3".
the printed name of the a82r3 is "Room 3".
Understand "Area 82 in Room 3" as a82r3.
the parent of the a82r3 is r3.
the X of the a82r3 is "548022.005982906".
the Y of the a82r3 is "1678505.7505982907".

[create a83r3]
a83r3 is a area. "An area (83) in r3".
the printed name of the a83r3 is "Room 3".
Understand "Area 83 in Room 3" as a83r3.
the parent of the a83r3 is r3.
the X of the a83r3 is "549063.0".
the Y of the a83r3 is "1678523.0".

[create a84r3]
a84r3 is a area. "An area (84) in r3".
the printed name of the a84r3 is "Room 3".
Understand "Area 84 in Room 3" as a84r3.
the parent of the a84r3 is r3.
the X of the a84r3 is "549032.037202381".
the Y of the a84r3 is "1678513.5".

[create a90r3]
a90r3 is a area. "An area (90) in r3".
the printed name of the a90r3 is "Room 3".
Understand "Area 90 in Room 3" as a90r3.
the parent of the a90r3 is r3.
the X of the a90r3 is "550106.8043478262".
the Y of the a90r3 is "1678375.0".

[create a1r4]
a1r4 is a area. "An area (1) in r4".
the printed name of the a1r4 is "Room 4".
Understand "Area 1 in Room 4" as a1r4.
the parent of the a1r4 is r4.
the X of the a1r4 is "548919.7233230514".
the Y of the a1r4 is "1677210.8028498502".

[create a18r4]
a18r4 is a area. "An area (18) in r4".
the printed name of the a18r4 is "Room 4".
Understand "Area 18 in Room 4" as a18r4.
the parent of the a18r4 is r4.
the X of the a18r4 is "548938.1865194305".
the Y of the a18r4 is "1676570.3208574706".

[create a19r4]
a19r4 is a area. "An area (19) in r4".
the printed name of the a19r4 is "Room 4".
Understand "Area 19 in Room 4" as a19r4.
the parent of the a19r4 is r4.
the X of the a19r4 is "547994.0".
the Y of the a19r4 is "1676568.5778301887".

[create a20r4]
a20r4 is a area. "An area (20) in r4".
the printed name of the a20r4 is "Room 4".
Understand "Area 20 in Room 4" as a20r4.
the parent of the a20r4 is r4.
the X of the a20r4 is "550438.0324175665".
the Y of the a20r4 is "1677102.013516861".

[create a22r4]
a22r4 is a area. "An area (22) in r4".
the printed name of the a22r4 is "Room 4".
Understand "Area 22 in Room 4" as a22r4.
the parent of the a22r4 is r4.
the X of the a22r4 is "548360.7817693062".
the Y of the a22r4 is "1676444.9748289345".

[create a23r4]
a23r4 is a area. "An area (23) in r4".
the printed name of the a23r4 is "Room 4".
Understand "Area 23 in Room 4" as a23r4.
the parent of the a23r4 is r4.
the X of the a23r4 is "547612.3891992549".
the Y of the a23r4 is "1676440.2597765361".

[create a24r4]
a24r4 is a area. "An area (24) in r4".
the printed name of the a24r4 is "Room 4".
Understand "Area 24 in Room 4" as a24r4.
the parent of the a24r4 is r4.
the X of the a24r4 is "548554.3891992552".
the Y of the a24r4 is "1676440.2597765361".

[create a25r4]
a25r4 is a area. "An area (25) in r4".
the printed name of the a25r4 is "Room 4".
Understand "Area 25 in Room 4" as a25r4.
the parent of the a25r4 is r4.
the X of the a25r4 is "549957.4184328843".
the Y of the a25r4 is "1676536.1969348688".

[create a57r4]
a57r4 is a area. "An area (57) in r4".
the printed name of the a57r4 is "Room 4".
Understand "Area 57 in Room 4" as a57r4.
the parent of the a57r4 is r4.
the X of the a57r4 is "547558.0".
the Y of the a57r4 is "1676480.0".

[create a58r4]
a58r4 is a area. "An area (58) in r4".
the printed name of the a58r4 is "Room 4".
Understand "Area 58 in Room 4" as a58r4.
the parent of the a58r4 is r4.
the X of the a58r4 is "547589.0059757691".
the Y of the a58r4 is "1676460.657907662".

[create a59r4]
a59r4 is a area. "An area (59) in r4".
the printed name of the a59r4 is "Room 4".
Understand "Area 59 in Room 4" as a59r4.
the parent of the a59r4 is r4.
the X of the a59r4 is "548409.0".
the Y of the a59r4 is "1676391.0".

[create a60r4]
a60r4 is a area. "An area (60) in r4".
the printed name of the a60r4 is "Room 4".
Understand "Area 60 in Room 4" as a60r4.
the parent of the a60r4 is r4.
the X of the a60r4 is "548377.7862838916".
the Y of the a60r4 is "1676416.633971292".

[create a61r4]
a61r4 is a area. "An area (61) in r4".
the printed name of the a61r4 is "Room 4".
Understand "Area 61 in Room 4" as a61r4.
the parent of the a61r4 is r4.
the X of the a61r4 is "548418.0".
the Y of the a61r4 is "1676480.0".

[create a62r4]
a62r4 is a area. "An area (62) in r4".
the printed name of the a62r4 is "Room 4".
Understand "Area 62 in Room 4" as a62r4.
the parent of the a62r4 is r4.
the X of the a62r4 is "548375.1485252037".
the Y of the a62r4 is "1676461.8557671146".

[create a63r4]
a63r4 is a area. "An area (63) in r4".
the printed name of the a63r4 is "Room 4".
Understand "Area 63 in Room 4" as a63r4.
the parent of the a63r4 is r4.
the X of the a63r4 is "550015.0".
the Y of the a63r4 is "1676400.0".

[create a64r4]
a64r4 is a area. "An area (64) in r4".
the printed name of the a64r4 is "Room 4".
Understand "Area 64 in Room 4" as a64r4.
the parent of the a64r4 is r4.
the X of the a64r4 is "549978.875".
the Y of the a64r4 is "1676485.720959596".

[create a65r4]
a65r4 is a area. "An area (65) in r4".
the printed name of the a65r4 is "Room 4".
Understand "Area 65 in Room 4" as a65r4.
the parent of the a65r4 is r4.
the X of the a65r4 is "550623.0".
the Y of the a65r4 is "1677027.0".

[create a66r4]
a66r4 is a area. "An area (66) in r4".
the printed name of the a66r4 is "Room 4".
Understand "Area 66 in Room 4" as a66r4.
the parent of the a66r4 is r4.
the X of the a66r4 is "550534.3368675595".
the Y of the a66r4 is "1677062.2857142857".

[create a67r4]
a67r4 is a area. "An area (67) in r4".
the printed name of the a67r4 is "Room 4".
Understand "Area 67 in Room 4" as a67r4.
the parent of the a67r4 is r4.
the X of the a67r4 is "548781.0".
the Y of the a67r4 is "1677110.0".

[create a68r4]
a68r4 is a area. "An area (68) in r4".
the printed name of the a68r4 is "Room 4".
Understand "Area 68 in Room 4" as a68r4.
the parent of the a68r4 is r4.
the X of the a68r4 is "548785.6828835822".
the Y of the a68r4 is "1677138.420352564".

[create a73r4]
a73r4 is a area. "An area (73) in r4".
the printed name of the a73r4 is "Room 4".
Understand "Area 73 in Room 4" as a73r4.
the parent of the a73r4 is r4.
the X of the a73r4 is "547567.0".
the Y of the a73r4 is "1676391.0".

[create a74r4]
a74r4 is a area. "An area (74) in r4".
the printed name of the a74r4 is "Room 4".
Understand "Area 74 in Room 4" as a74r4.
the parent of the a74r4 is r4.
the X of the a74r4 is "547598.2137161084".
the Y of the a74r4 is "1676416.633971292".

[create a75r4]
a75r4 is a area. "An area (75) in r4".
the printed name of the a75r4 is "Room 4".
Understand "Area 75 in Room 4" as a75r4.
the parent of the a75r4 is r4.
the X of the a75r4 is "548509.0".
the Y of the a75r4 is "1676391.0".

[create a76r4]
a76r4 is a area. "An area (76) in r4".
the printed name of the a76r4 is "Room 4".
Understand "Area 76 in Room 4" as a76r4.
the parent of the a76r4 is r4.
the X of the a76r4 is "548540.2137161084".
the Y of the a76r4 is "1676416.6339712918".

[create a77r4]
a77r4 is a area. "An area (77) in r4".
the printed name of the a77r4 is "Room 4".
Understand "Area 77 in Room 4" as a77r4.
the parent of the a77r4 is r4.
the X of the a77r4 is "548500.0".
the Y of the a77r4 is "1676480.0".

[create a78r4]
a78r4 is a area. "An area (78) in r4".
the printed name of the a78r4 is "Room 4".
Understand "Area 78 in Room 4" as a78r4.
the parent of the a78r4 is r4.
the X of the a78r4 is "548531.0059757695".
the Y of the a78r4 is "1676460.657907662".

[create a21r5]
a21r5 is a area. "An area (21) in r5".
the printed name of the a21r5 is "Room 5".
Understand "Area 21 in Room 5" as a21r5.
the parent of the a21r5 is r5.
the X of the a21r5 is "550322.4156343282".
the Y of the a21r5 is "1678238.8828591001".

[create a88r5]
a88r5 is a area. "An area (88) in r5".
the printed name of the a88r5 is "Room 5".
Understand "Area 88 in Room 5" as a88r5.
the parent of the a88r5 is r5.
the X of the a88r5 is "550300.0".
the Y of the a88r5 is "1678275.677419355".

[create a15r6]
a15r6 is a area. "An area (15) in r6".
the printed name of the a15r6 is "Room 6".
Understand "Area 15 in Room 6" as a15r6.
the parent of the a15r6 is r6.
the X of the a15r6 is "545405.7977073499".
the Y of the a15r6 is "1678344.5677680378".

[create a36r6]
a36r6 is a area. "An area (36) in r6".
the printed name of the a36r6 is "Room 6".
Understand "Area 36 in Room 6" as a36r6.
the parent of the a36r6 is r6.
the X of the a36r6 is "545050.2813718211".
the Y of the a36r6 is "1678461.2845894438".

[create a37r6]
a37r6 is a area. "An area (37) in r6".
the printed name of the a37r6 is "Room 6".
Understand "Area 37 in Room 6" as a37r6.
the parent of the a37r6 is r6.
the X of the a37r6 is "545779.1653756485".
the Y of the a37r6 is "1678461.2658517454".

[create a43r6]
a43r6 is a area. "An area (43) in r6".
the printed name of the a43r6 is "Room 6".
Understand "Area 43 in Room 6" as a43r6.
the parent of the a43r6 is r6.
the X of the a43r6 is "544994.0".
the Y of the a43r6 is "1678424.0".

[create a44r6]
a44r6 is a area. "An area (44) in r6".
the printed name of the a44r6 is "Room 6".
Understand "Area 44 in Room 6" as a44r6.
the parent of the a44r6 is r6.
the X of the a44r6 is "545033.4941364606".
the Y of the a44r6 is "1678442.7755863538".

[create a45r6]
a45r6 is a area. "An area (45) in r6".
the printed name of the a45r6 is "Room 6".
Understand "Area 45 in Room 6" as a45r6.
the parent of the a45r6 is r6.
the X of the a45r6 is "545836.0".
the Y of the a45r6 is "1678424.0".

[create a46r6]
a46r6 is a area. "An area (46) in r6".
the printed name of the a46r6 is "Room 6".
Understand "Area 46 in Room 6" as a46r6.
the parent of the a46r6 is r6.
the X of the a46r6 is "545795.6686930091".
the Y of the a46r6 is "1678442.2965214453".

[create a49r6]
a49r6 is a area. "An area (49) in r6".
the printed name of the a49r6 is "Room 6".
Understand "Area 49 in Room 6" as a49r6.
the parent of the a49r6 is r6.
the X of the a49r6 is "545826.0".
the Y of the a49r6 is "1678514.0".

[create a50r6]
a50r6 is a area. "An area (50) in r6".
the printed name of the a50r6 is "Room 6".
Understand "Area 50 in Room 6" as a50r6.
the parent of the a50r6 is r6.
the X of the a50r6 is "545795.1050372068".
the Y of the a50r6 is "1678487.7894485786".

[create a53r6]
a53r6 is a area. "An area (53) in r6".
the printed name of the a53r6 is "Room 6".
Understand "Area 53 in Room 6" as a53r6.
the parent of the a53r6 is r6.
the X of the a53r6 is "545003.0".
the Y of the a53r6 is "1678514.0".

[create a54r6]
a54r6 is a area. "An area (54) in r6".
the printed name of the a54r6 is "Room 6".
Understand "Area 54 in Room 6" as a54r6.
the parent of the a54r6 is r6.
the X of the a54r6 is "545034.0876068377".
the Y of the a54r6 is "1678488.1662393163".

[create a7r7]
a7r7 is a area. "An area (7) in r7".
the printed name of the a7r7 is "Room 7".
Understand "Area 7 in Room 7" as a7r7.
the parent of the a7r7 is r7.
the X of the a7r7 is "550261.8286071355".
the Y of the a7r7 is "1678342.424978698".

[create a8r7]
a8r7 is a area. "An area (8) in r7".
the printed name of the a8r7 is "Room 7".
Understand "Area 8 in Room 7" as a8r7.
the parent of the a8r7 is r7.
the X of the a8r7 is "550284.6291442799".
the Y of the a8r7 is "1678300.9081966185".

[create a9r7]
a9r7 is a area. "An area (9) in r7".
the printed name of the a9r7 is "Room 7".
Understand "Area 9 in Room 7" as a9r7.
the parent of the a9r7 is r7.
the X of the a9r7 is "550203.402173913".
the Y of the a9r7 is "1678363.552540156".

[create a39r7]
a39r7 is a area. "An area (39) in r7".
the printed name of the a39r7 is "Room 7".
Understand "Area 39 in Room 7" as a39r7.
the parent of the a39r7 is r7.
the X of the a39r7 is "550300.0".
the Y of the a39r7 is "1678375.0".

[create a40r7]
a40r7 is a area. "An area (40) in r7".
the printed name of the a40r7 is "Room 7".
Understand "Area 40 in Room 7" as a40r7.
the parent of the a40r7 is r7.
the X of the a40r7 is "550267.8007246378".
the Y of the a40r7 is "1678350.1693548388".

[create a88r7]
a88r7 is a area. "An area (88) in r7".
the printed name of the a88r7 is "Room 7".
Understand "Area 88 in Room 7" as a88r7.
the parent of the a88r7 is r7.
the X of the a88r7 is "550300.0".
the Y of the a88r7 is "1678275.677419355".

[create a89r7]
a89r7 is a area. "An area (89) in r7".
the printed name of the a89r7 is "Room 7".
Understand "Area 89 in Room 7" as a89r7.
the parent of the a89r7 is r7.
the X of the a89r7 is "550125.0".
the Y of the a89r7 is "1678366.0".

[create a90r7]
a90r7 is a area. "An area (90) in r7".
the printed name of the a90r7 is "Room 7".
Understand "Area 90 in Room 7" as a90r7.
the parent of the a90r7 is r7.
the X of the a90r7 is "550106.8043478262".
the Y of the a90r7 is "1678375.0".

[create d0]
d0 is a door. "A door between a10r2 and a12r3".
d0 is north of a10r2 and south of a12r3.

[create d2]
d2 is a door. "A door between a13r0 and a14r1".
d2 is north of a13r0 and south of a14r1.

[create d4]
d4 is a door. "A door between a16r0 and a15r6".
d4 is north of a16r0 and south of a15r6.

[create d6]
d6 is a door. "A door between a14r1 and a17r2".
d6 is southeast of a14r1 and northwest of a17r2.

[create d8]
d8 is a door. "A door between a0r3 and a1r4".
d8 is south of a0r3 and north of a1r4.

[create d9]
d9 is a door. "A door between a2r3 and a19r4".
d9 is south of a2r3 and north of a19r4.

[create d10]
d10 is a door. "A door between a20r4 and a21r5".
d10 is north of a20r4 and south of a21r5.

west of a0r3 is east of a2r3.

north of a2r3 is south of a3r3.

north of a0r3 is south of a4r3.

north of a4r3 is south of a5r3.

north of a5r3 is south of a6r3.

southeast of a7r7 is northwest of a8r7.

west of a7r7 is east of a9r7.

southeast of a10r2 is northwest of a11r2.

south of a1r4 is north of a18r4.

east of a19r4 is west of a22r4.

west of a19r4 is east of a23r4.

west of a18r4 is east of a24r4.

east of a18r4 is west of a25r4.

southwest of a16r0 is northeast of a26r0.

southwest of a17r2 is northeast of a27r2.

southwest of a20r4 is northeast of a25r4.

southeast of a13r0 is northwest of a28r0.

west of a3r3 is east of a12r3.

northeast of a6r3 is southwest of a29r3.

north of a14r1 is south of a30r1.

east of a29r3 is west of a31r3.

west of a6r3 is east of a32r3.

east of a33r3 is west of a34r3.

northwest of a30r1 is southeast of a35r1.

west of a15r6 is east of a36r6.

east of a15r6 is west of a37r6.

west of a32r3 is east of a38r3.

southwest of a39r7 is northeast of a40r7.

northeast of a41r3 is southwest of a42r3.

northeast of a43r6 is southwest of a44r6.

northwest of a45r6 is southeast of a46r6.

southwest of a47r1 is northeast of a48r1.

southwest of a49r6 is northeast of a50r6.

southeast of a51r3 is northwest of a52r3.

southeast of a53r6 is northwest of a54r6.

east of a55r3 is west of a56r3.

southeast of a57r4 is northwest of a58r4.

northwest of a59r4 is southeast of a60r4.

southwest of a61r4 is northeast of a62r4.

northwest of a63r4 is southeast of a64r4.

west of a65r4 is east of a66r4.

north of a67r4 is south of a68r4.

south of a69r3 is north of a70r3.

northwest of a71r3 is southeast of a72r3.

northeast of a73r4 is southwest of a74r4.

northeast of a75r4 is southwest of a76r4.

southeast of a77r4 is northwest of a78r4.

east of a79r3 is west of a80r3.

west of a81r3 is east of a82r3.

west of a83r3 is east of a84r3.

southwest of a27r2 is northeast of a85r2.

southeast of a28r0 is northwest of a86r0.

southeast of a11r2 is northwest of a87r2.

northwest of a21r5 is southeast of a88r5.

northwest of a88r5 is southeast of a8r7.

west of a9r7 is east of a89r7.

east of a31r3 is west of a90r3.

northwest of a35r1 is southeast of a91r1.

southwest of a26r0 is northeast of a92r0.

northwest of a33r3 is southeast of a52r3.

northwest of a36r6 is southeast of a54r6.

north of a32r3 is south of a56r3.

north of a34r3 is south of a80r3.

north of a38r3 is south of a82r3.

north of a29r3 is south of a84r3.

northeast of a7r7 is southwest of a40r7.

northeast of a3r3 is southwest of a38r3.

northwest of a12r3 is southeast of a34r3.

southwest of a33r3 is northeast of a42r3.

southwest of a36r6 is northeast of a44r6.

southeast of a37r6 is northwest of a46r6.

northeast of a30r1 is southwest of a48r1.

northeast of a37r6 is southwest of a50r6.

northeast of a22r4 is southwest of a62r4.

southeast of a25r4 is northwest of a64r4.

west of a10r2 is east of a17r2.

west of a13r0 is east of a16r0.

east of a20r4 is west of a66r4.

southwest of a1r4 is northeast of a68r4.

northwest of a4r3 is southeast of a70r3.

east of a5r3 is west of a72r3.

southwest of a23r4 is northeast of a74r4.

southwest of a24r4 is northeast of a76r4.

northwest of a24r4 is southeast of a78r4.

northwest of a23r4 is southeast of a58r4.

southeast of a22r4 is northwest of a60r4.

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
the player is in a13r0.

the orientation of the player is 6.
