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

[create r8]
r8 is a indoor_room. "Room 8".

[create a2r0]
a2r0 is a area. "An area (2) in r0".
the printed name of the a2r0 is "Room 0".
Understand "Area 2 in Room 0" as a2r0.
the parent of the a2r0 is r0.
the X of the a2r0 is "545786.8816727516".
the Y of the a2r0 is "1706819.4952958585".

[create a3r0]
a3r0 is a area. "An area (3) in r0".
the printed name of the a3r0 is "Room 0".
Understand "Area 3 in Room 0" as a3r0.
the parent of the a3r0 is r0.
the X of the a3r0 is "545890.0086022713".
the Y of the a3r0 is "1706964.0104014394".

[create a28r0]
a28r0 is a area. "An area (28) in r0".
the printed name of the a28r0 is "Room 0".
Understand "Area 28 in Room 0" as a28r0.
the parent of the a28r0 is r0.
the X of the a28r0 is "545903.7496886394".
the Y of the a28r0 is "1706962.317373606".

[create a29r0]
a29r0 is a area. "An area (29) in r0".
the printed name of the a29r0 is "Room 0".
Understand "Area 29 in Room 0" as a29r0.
the parent of the a29r0 is r0.
the X of the a29r0 is "545997.2968536066".
the Y of the a29r0 is "1706851.4209743899".

[create a67r0]
a67r0 is a area. "An area (67) in r0".
the printed name of the a67r0 is "Room 0".
Understand "Area 67 in Room 0" as a67r0.
the parent of the a67r0 is r0.
the X of the a67r0 is "546038.0".
the Y of the a67r0 is "1706800.0".

[create a73r0]
a73r0 is a area. "An area (73) in r0".
the printed name of the a73r0 is "Room 0".
Understand "Area 73 in Room 0" as a73r0.
the parent of the a73r0 is r0.
the X of the a73r0 is "545756.0".
the Y of the a73r0 is "1706772.0".

[create a12r1]
a12r1 is a area. "An area (12) in r1".
the printed name of the a12r1 is "Room 1".
Understand "Area 12 in Room 1" as a12r1.
the parent of the a12r1 is r1.
the X of the a12r1 is "546477.3333333333".
the Y of the a12r1 is "1708860.9694444444".

[create a13r1]
a13r1 is a area. "An area (13) in r1".
the printed name of the a13r1 is "Room 1".
Understand "Area 13 in Room 1" as a13r1.
the parent of the a13r1 is r1.
the X of the a13r1 is "546544.5167685037".
the Y of the a13r1 is "1708773.513395759".

[create a24r1]
a24r1 is a area. "An area (24) in r1".
the printed name of the a24r1 is "Room 1".
Understand "Area 24 in Room 1" as a24r1.
the parent of the a24r1 is r1.
the X of the a24r1 is "546549.8708448356".
the Y of the a24r1 is "1708605.3162913686".

[create a64r1]
a64r1 is a area. "An area (64) in r1".
the printed name of the a64r1 is "Room 1".
Understand "Area 64 in Room 1" as a64r1.
the parent of the a64r1 is r1.
the X of the a64r1 is "546644.0".
the Y of the a64r1 is "1708904.0".

[create a65r1]
a65r1 is a area. "An area (65) in r1".
the printed name of the a65r1 is "Room 1".
Understand "Area 65 in Room 1" as a65r1.
the parent of the a65r1 is r1.
the X of the a65r1 is "546610.6666666666".
the Y of the a65r1 is "1708853.5364495006".

[create a70r1]
a70r1 is a area. "An area (70) in r1".
the printed name of the a70r1 is "Room 1".
Understand "Area 70 in Room 1" as a70r1.
the parent of the a70r1 is r1.
the X of the a70r1 is "546444.0".
the Y of the a70r1 is "1708904.0".

[create a4r2]
a4r2 is a area. "An area (4) in r2".
the printed name of the a4r2 is "Room 2".
Understand "Area 4 in Room 2" as a4r2.
the parent of the a4r2 is r2.
the X of the a4r2 is "547048.7043713275".
the Y of the a4r2 is "1706822.706711225".

[create a5r2]
a5r2 is a area. "An area (5) in r2".
the printed name of the a5r2 is "Room 2".
Understand "Area 5 in Room 2" as a5r2.
the parent of the a5r2 is r2.
the X of the a5r2 is "547144.2390740267".
the Y of the a5r2 is "1706962.431282769".

[create a32r2]
a32r2 is a area. "An area (32) in r2".
the printed name of the a32r2 is "Room 2".
Understand "Area 32 in Room 2" as a32r2.
the parent of the a32r2 is r2.
the X of the a32r2 is "547166.289651646".
the Y of the a32r2 is "1706960.8833195833".

[create a33r2]
a33r2 is a area. "An area (33) in r2".
the printed name of the a33r2 is "Room 2".
Understand "Area 33 in Room 2" as a33r2.
the parent of the a33r2 is r2.
the X of the a33r2 is "547260.4539705468".
the Y of the a33r2 is "1706848.7643334027".

[create a69r2]
a69r2 is a area. "An area (69) in r2".
the printed name of the a69r2 is "Room 2".
Understand "Area 69 in Room 2" as a69r2.
the parent of the a69r2 is r2.
the X of the a69r2 is "547301.0".
the Y of the a69r2 is "1706800.0".

[create a74r2]
a74r2 is a area. "An area (74) in r2".
the printed name of the a74r2 is "Room 2".
Understand "Area 74 in Room 2" as a74r2.
the parent of the a74r2 is r2.
the X of the a74r2 is "547018.0".
the Y of the a74r2 is "1706772.0".

[create a8r3]
a8r3 is a area. "An area (8) in r3".
the printed name of the a8r3 is "Room 3".
Understand "Area 8 in Room 3" as a8r3.
the parent of the a8r3 is r3.
the X of the a8r3 is "547341.9147631298".
the Y of the a8r3 is "1708860.1324189957".

[create a9r3]
a9r3 is a area. "An area (9) in r3".
the printed name of the a9r3 is "Room 3".
Understand "Area 9 in Room 3" as a9r3.
the parent of the a9r3 is r3.
the X of the a9r3 is "547509.6102763403".
the Y of the a9r3 is "1708842.8408194105".

[create a10r3]
a10r3 is a area. "An area (10) in r3".
the printed name of the a10r3 is "Room 3".
Understand "Area 10 in Room 3" as a10r3.
the parent of the a10r3 is r3.
the X of the a10r3 is "547722.4094539563".
the Y of the a10r3 is "1708730.9460312452".

[create a11r3]
a11r3 is a area. "An area (11) in r3".
the printed name of the a11r3 is "Room 3".
Understand "Area 11 in Room 3" as a11r3.
the parent of the a11r3 is r3.
the X of the a11r3 is "547850.3900436837".
the Y of the a11r3 is "1708720.4008683986".

[create a16r3]
a16r3 is a area. "An area (16) in r3".
the printed name of the a16r3 is "Room 3".
Understand "Area 16 in Room 3" as a16r3.
the parent of the a16r3 is r3.
the X of the a16r3 is "548014.0421155228".
the Y of the a16r3 is "1708827.3065045658".

[create a17r3]
a17r3 is a area. "An area (17) in r3".
the printed name of the a17r3 is "Room 3".
Understand "Area 17 in Room 3" as a17r3.
the parent of the a17r3 is r3.
the X of the a17r3 is "548248.1653756485".
the Y of the a17r3 is "1708861.2658517451".

[create a34r3]
a34r3 is a area. "An area (34) in r3".
the printed name of the a34r3 is "Room 3".
Understand "Area 34 in Room 3" as a34r3.
the parent of the a34r3 is r3.
the X of the a34r3 is "547471.087628866".
the Y of the a34r3 is "1708904.8762886599".

[create a35r3]
a35r3 is a area. "An area (35) in r3".
the printed name of the a35r3 is "Room 3".
Understand "Area 35 in Room 3" as a35r3.
the parent of the a35r3 is r3.
the X of the a35r3 is "547504.0167107234".
the Y of the a35r3 is "1708906.1908547007".

[create a38r3]
a38r3 is a area. "An area (38) in r3".
the printed name of the a38r3 is "Room 3".
Understand "Area 38 in Room 3" as a38r3.
the parent of the a38r3 is r3.
the X of the a38r3 is "547295.0".
the Y of the a38r3 is "1708914.0".

[create a39r3]
a39r3 is a area. "An area (39) in r3".
the printed name of the a39r3 is "Room 3".
Understand "Area 39 in Room 3" as a39r3.
the parent of the a39r3 is r3.
the X of the a39r3 is "547323.1036507601".
the Y of the a39r3 is "1708888.4646349237".

[create a42r3]
a42r3 is a area. "An area (42) in r3".
the printed name of the a42r3 is "Room 3".
Understand "Area 42 in Room 3" as a42r3.
the parent of the a42r3 is r3.
the X of the a42r3 is "548295.0".
the Y of the a42r3 is "1708914.0".

[create a43r3]
a43r3 is a area. "An area (43) in r3".
the printed name of the a43r3 is "Room 3".
Understand "Area 43 in Room 3" as a43r3.
the parent of the a43r3 is r3.
the X of the a43r3 is "548264.1050372066".
the Y of the a43r3 is "1708887.7894485784".

[create a50r3]
a50r3 is a area. "An area (50) in r3".
the printed name of the a50r3 is "Room 3".
Understand "Area 50 in Room 3" as a50r3.
the parent of the a50r3 is r3.
the X of the a50r3 is "548055.0".
the Y of the a50r3 is "1708904.0".

[create a51r3]
a51r3 is a area. "An area (51) in r3".
the printed name of the a51r3 is "Room 3".
Understand "Area 51 in Room 3" as a51r3.
the parent of the a51r3 is r3.
the X of the a51r3 is "548022.0059829061".
the Y of the a51r3 is "1708905.7505982905".

[create a52r3]
a52r3 is a area. "An area (52) in r3".
the printed name of the a52r3 is "Room 3".
Understand "Area 52 in Room 3" as a52r3.
the parent of the a52r3 is r3.
the X of the a52r3 is "547286.0".
the Y of the a52r3 is "1708824.0".

[create a53r3]
a53r3 is a area. "An area (53) in r3".
the printed name of the a53r3 is "Room 3".
Understand "Area 53 in Room 3" as a53r3.
the parent of the a53r3 is r3.
the X of the a53r3 is "547326.7622699386".
the Y of the a53r3 is "1708842.6487730062".

[create a62r3]
a62r3 is a area. "An area (62) in r3".
the printed name of the a62r3 is "Room 3".
Understand "Area 62 in Room 3" as a62r3.
the parent of the a62r3 is r3.
the X of the a62r3 is "548305.0".
the Y of the a62r3 is "1708824.0".

[create a63r3]
a63r3 is a area. "An area (63) in r3".
the printed name of the a63r3 is "Room 3".
Understand "Area 63 in Room 3" as a63r3.
the parent of the a63r3 is r3.
the X of the a63r3 is "548264.6686930091".
the Y of the a63r3 is "1708842.2965214455".

[create a0r4]
a0r4 is a area. "An area (0) in r4".
the printed name of the a0r4 is "Room 4".
Understand "Area 0 in Room 4" as a0r4.
the parent of the a0r4 is r4.
the X of the a0r4 is "548255.8448489627".
the Y of the a0r4 is "1706819.8661640184".

[create a1r4]
a1r4 is a area. "An area (1) in r4".
the printed name of the a1r4 is "Room 4".
Understand "Area 1 in Room 4" as a1r4.
the parent of the a1r4 is r4.
the X of the a1r4 is "548358.9979687049".
the Y of the a1r4 is "1706964.011711592".

[create a22r4]
a22r4 is a area. "An area (22) in r4".
the printed name of the a22r4 is "Room 4".
Understand "Area 22 in Room 4" as a22r4.
the parent of the a22r4 is r4.
the X of the a22r4 is "548372.7496886395".
the Y of the a22r4 is "1706962.317373606".

[create a27r4]
a27r4 is a area. "An area (27) in r4".
the printed name of the a27r4 is "Room 4".
Understand "Area 27 in Room 4" as a27r4.
the parent of the a27r4 is r4.
the X of the a27r4 is "548466.2968536066".
the Y of the a27r4 is "1706851.4209743899".

[create a66r4]
a66r4 is a area. "An area (66) in r4".
the printed name of the a66r4 is "Room 4".
Understand "Area 66 in Room 4" as a66r4.
the parent of the a66r4 is r4.
the X of the a66r4 is "548507.0".
the Y of the a66r4 is "1706800.0".

[create a72r4]
a72r4 is a area. "An area (72) in r4".
the printed name of the a72r4 is "Room 4".
Understand "Area 72 in Room 4" as a72r4.
the parent of the a72r4 is r4.
the X of the a72r4 is "548225.0".
the Y of the a72r4 is "1706772.0".

[create a14r5]
a14r5 is a area. "An area (14) in r5".
the printed name of the a14r5 is "Room 5".
Understand "Area 14 in Room 5" as a14r5.
the parent of the a14r5 is r5.
the X of the a14r5 is "548946.3333333333".
the Y of the a14r5 is "1708860.9694444444".

[create a15r5]
a15r5 is a area. "An area (15) in r5".
the printed name of the a15r5 is "Room 5".
Understand "Area 15 in Room 5" as a15r5.
the parent of the a15r5 is r5.
the X of the a15r5 is "549013.5167685037".
the Y of the a15r5 is "1708773.513395759".

[create a23r5]
a23r5 is a area. "An area (23) in r5".
the printed name of the a23r5 is "Room 5".
Understand "Area 23 in Room 5" as a23r5.
the parent of the a23r5 is r5.
the X of the a23r5 is "549018.8708448359".
the Y of the a23r5 is "1708605.3162913686".

[create a36r5]
a36r5 is a area. "An area (36) in r5".
the printed name of the a36r5 is "Room 5".
Understand "Area 36 in Room 5" as a36r5.
the parent of the a36r5 is r5.
the X of the a36r5 is "549113.0".
the Y of the a36r5 is "1708904.0".

[create a37r5]
a37r5 is a area. "An area (37) in r5".
the printed name of the a37r5 is "Room 5".
Understand "Area 37 in Room 5" as a37r5.
the parent of the a37r5 is r5.
the X of the a37r5 is "549079.6666666665".
the Y of the a37r5 is "1708853.5364495006".

[create a71r5]
a71r5 is a area. "An area (71) in r5".
the printed name of the a71r5 is "Room 5".
Understand "Area 71 in Room 5" as a71r5.
the parent of the a71r5 is r5.
the X of the a71r5 is "548913.0".
the Y of the a71r5 is "1708904.0".

[create a25r6]
a25r6 is a area. "An area (25) in r6".
the printed name of the a25r6 is "Room 6".
Understand "Area 25 in Room 6" as a25r6.
the parent of the a25r6 is r6.
the X of the a25r6 is "549613.2390740267".
the Y of the a25r6 is "1706962.431282769".

[create a26r6]
a26r6 is a area. "An area (26) in r6".
the printed name of the a26r6 is "Room 6".
Understand "Area 26 in Room 6" as a26r6.
the parent of the a26r6 is r6.
the X of the a26r6 is "549517.7043713275".
the Y of the a26r6 is "1706822.7067112252".

[create a30r6]
a30r6 is a area. "An area (30) in r6".
the printed name of the a30r6 is "Room 6".
Understand "Area 30 in Room 6" as a30r6.
the parent of the a30r6 is r6.
the X of the a30r6 is "549635.2896516463".
the Y of the a30r6 is "1706960.8833195835".

[create a31r6]
a31r6 is a area. "An area (31) in r6".
the printed name of the a31r6 is "Room 6".
Understand "Area 31 in Room 6" as a31r6.
the parent of the a31r6 is r6.
the X of the a31r6 is "549729.4539705468".
the Y of the a31r6 is "1706848.7643334027".

[create a68r6]
a68r6 is a area. "An area (68) in r6".
the printed name of the a68r6 is "Room 6".
Understand "Area 68 in Room 6" as a68r6.
the parent of the a68r6 is r6.
the X of the a68r6 is "549770.0".
the Y of the a68r6 is "1706800.0".

[create a75r6]
a75r6 is a area. "An area (75) in r6".
the printed name of the a75r6 is "Room 6".
Understand "Area 75 in Room 6" as a75r6.
the parent of the a75r6 is r6.
the X of the a75r6 is "549487.0".
the Y of the a75r6 is "1706772.0".

[create a6r7]
a6r7 is a area. "An area (6) in r7".
the printed name of the a6r7 is "Room 7".
Understand "Area 6 in Room 7" as a6r7.
the parent of the a6r7 is r7.
the X of the a6r7 is "550145.9486065779".
the Y of the a6r7 is "1708778.1064398694".

[create a7r7]
a7r7 is a area. "An area (7) in r7".
the printed name of the a7r7 is "Room 7".
Understand "Area 7 in Room 7" as a7r7.
the parent of the a7r7 is r7.
the X of the a7r7 is "550475.5509290382".
the Y of the a7r7 is "1708861.0062088708".

[create a21r7]
a21r7 is a area. "An area (21) in r7".
the printed name of the a21r7 is "Room 7".
Understand "Area 21 in Room 7" as a21r7.
the parent of the a21r7 is r7.
the X of the a21r7 is "549811.4921500897".
the Y of the a21r7 is "1708860.7986347186".

[create a44r7]
a44r7 is a area. "An area (44) in r7".
the printed name of the a44r7 is "Room 7".
Understand "Area 44 in Room 7" as a44r7.
the parent of the a44r7 is r7.
the X of the a44r7 is "549764.0".
the Y of the a44r7 is "1708914.0".

[create a45r7]
a45r7 is a area. "An area (45) in r7".
the printed name of the a45r7 is "Room 7".
Understand "Area 45 in Room 7" as a45r7.
the parent of the a45r7 is r7.
the X of the a45r7 is "549794.9153780069".
the Y of the a45r7 is "1708888.1834621995".

[create a48r7]
a48r7 is a area. "An area (48) in r7".
the printed name of the a48r7 is "Room 7".
Understand "Area 48 in Room 7" as a48r7.
the parent of the a48r7 is r7.
the X of the a48r7 is "550523.0".
the Y of the a48r7 is "1708914.0".

[create a49r7]
a49r7 is a area. "An area (49) in r7".
the printed name of the a49r7 is "Room 7".
Understand "Area 49 in Room 7" as a49r7.
the parent of the a49r7 is r7.
the X of the a49r7 is "550491.9123931625".
the Y of the a49r7 is "1708888.1662393163".

[create a54r7]
a54r7 is a area. "An area (54) in r7".
the printed name of the a54r7 is "Room 7".
Understand "Area 54 in Room 7" as a54r7.
the parent of the a54r7 is r7.
the X of the a54r7 is "549755.0".
the Y of the a54r7 is "1708824.0".

[create a55r7]
a55r7 is a area. "An area (55) in r7".
the printed name of the a55r7 is "Room 7".
Understand "Area 55 in Room 7" as a55r7.
the parent of the a55r7 is r7.
the X of the a55r7 is "549795.7622699386".
the Y of the a55r7 is "1708842.6487730062".

[create a56r7]
a56r7 is a area. "An area (56) in r7".
the printed name of the a56r7 is "Room 7".
Understand "Area 56 in Room 7" as a56r7.
the parent of the a56r7 is r7.
the X of the a56r7 is "550532.0".
the Y of the a56r7 is "1708824.0".

[create a57r7]
a57r7 is a area. "An area (57) in r7".
the printed name of the a57r7 is "Room 7".
Understand "Area 57 in Room 7" as a57r7.
the parent of the a57r7 is r7.
the X of the a57r7 is "550491.662577747".
the Y of the a57r7 is "1708842.6912577746".

[create a18r8]
a18r8 is a area. "An area (18) in r8".
the printed name of the a18r8 is "Room 8".
Understand "Area 18 in Room 8" as a18r8.
the parent of the a18r8 is r8.
the X of the a18r8 is "545050.2813718208".
the Y of the a18r8 is "1708861.2845894436".

[create a19r8]
a19r8 is a area. "An area (19) in r8".
the printed name of the a19r8 is "Room 8".
Understand "Area 19 in Room 8" as a19r8.
the parent of the a19r8 is r8.
the X of the a19r8 is "545405.79770735".
the Y of the a19r8 is "1708744.567768038".

[create a20r8]
a20r8 is a area. "An area (20) in r8".
the printed name of the a20r8 is "Room 8".
Understand "Area 20 in Room 8" as a20r8.
the parent of the a20r8 is r8.
the X of the a20r8 is "545779.1653756488".
the Y of the a20r8 is "1708861.2658517451".

[create a40r8]
a40r8 is a area. "An area (40) in r8".
the printed name of the a40r8 is "Room 8".
Understand "Area 40 in Room 8" as a40r8.
the parent of the a40r8 is r8.
the X of the a40r8 is "545826.0".
the Y of the a40r8 is "1708914.0".

[create a41r8]
a41r8 is a area. "An area (41) in r8".
the printed name of the a41r8 is "Room 8".
Understand "Area 41 in Room 8" as a41r8.
the parent of the a41r8 is r8.
the X of the a41r8 is "545795.1050372068".
the Y of the a41r8 is "1708887.7894485786".

[create a46r8]
a46r8 is a area. "An area (46) in r8".
the printed name of the a46r8 is "Room 8".
Understand "Area 46 in Room 8" as a46r8.
the parent of the a46r8 is r8.
the X of the a46r8 is "545003.0".
the Y of the a46r8 is "1708914.0".

[create a47r8]
a47r8 is a area. "An area (47) in r8".
the printed name of the a47r8 is "Room 8".
Understand "Area 47 in Room 8" as a47r8.
the parent of the a47r8 is r8.
the X of the a47r8 is "545034.0876068375".
the Y of the a47r8 is "1708888.166239316".

[create a58r8]
a58r8 is a area. "An area (58) in r8".
the printed name of the a58r8 is "Room 8".
Understand "Area 58 in Room 8" as a58r8.
the parent of the a58r8 is r8.
the X of the a58r8 is "544994.0".
the Y of the a58r8 is "1708824.0".

[create a59r8]
a59r8 is a area. "An area (59) in r8".
the printed name of the a59r8 is "Room 8".
Understand "Area 59 in Room 8" as a59r8.
the parent of the a59r8 is r8.
the X of the a59r8 is "545033.4941364606".
the Y of the a59r8 is "1708842.7755863538".

[create a60r8]
a60r8 is a area. "An area (60) in r8".
the printed name of the a60r8 is "Room 8".
Understand "Area 60 in Room 8" as a60r8.
the parent of the a60r8 is r8.
the X of the a60r8 is "545836.0".
the Y of the a60r8 is "1708824.0".

[create a61r8]
a61r8 is a area. "An area (61) in r8".
the printed name of the a61r8 is "Room 8".
Understand "Area 61 in Room 8" as a61r8.
the parent of the a61r8 is r8.
the X of the a61r8 is "545795.6686930091".
the Y of the a61r8 is "1708842.2965214455".

[create d0]
d0 is a door. "A door between a32r2 and a10r3".
d0 is north of a32r2 and south of a10r3.

[create d2]
d2 is a door. "A door between a30r6 and a6r7".
d2 is north of a30r6 and south of a6r7.

[create d4]
d4 is a door. "A door between a28r0 and a24r1".
d4 is north of a28r0 and south of a24r1.

[create d6]
d6 is a door. "A door between a22r4 and a23r5".
d6 is north of a22r4 and south of a23r5.

[create d8]
d8 is a door. "A door between a3r0 and a19r8".
d8 is north of a3r0 and south of a19r8.

[create d10]
d10 is a door. "A door between a11r3 and a1r4".
d10 is south of a11r3 and north of a1r4.

[create d12]
d12 is a door. "A door between a24r1 and a5r2".
d12 is southeast of a24r1 and northwest of a5r2.

[create d14]
d14 is a door. "A door between a23r5 and a25r6".
d14 is southeast of a23r5 and northwest of a25r6.

northeast of a0r4 is southwest of a1r4.

northeast of a2r0 is southwest of a3r0.

northeast of a4r2 is southwest of a5r2.

east of a6r7 is west of a7r7.

east of a8r3 is west of a9r3.

east of a10r3 is west of a11r3.

southeast of a9r3 is northwest of a10r3.

southeast of a12r1 is northwest of a13r1.

southeast of a14r5 is northwest of a15r5.

east of a16r3 is west of a17r3.

east of a18r8 is west of a19r8.

east of a19r8 is west of a20r8.

west of a6r7 is east of a21r7.

south of a13r1 is north of a24r1.

south of a15r5 is north of a23r5.

northeast of a11r3 is southwest of a16r3.

southwest of a25r6 is northeast of a26r6.

southeast of a22r4 is northwest of a27r4.

southeast of a28r0 is northwest of a29r0.

southeast of a30r6 is northwest of a31r6.

southeast of a32r2 is northwest of a33r2.

east of a34r3 is west of a35r3.

southwest of a36r5 is northeast of a37r5.

southeast of a38r3 is northwest of a39r3.

southwest of a40r8 is northeast of a41r8.

southwest of a42r3 is northeast of a43r3.

southeast of a44r7 is northwest of a45r7.

southeast of a46r8 is northwest of a47r8.

southwest of a48r7 is northeast of a49r7.

west of a50r3 is east of a51r3.

northeast of a52r3 is southwest of a53r3.

northeast of a54r7 is southwest of a55r7.

northwest of a56r7 is southeast of a57r7.

northeast of a58r8 is southwest of a59r8.

northwest of a60r8 is southeast of a61r8.

northwest of a62r3 is southeast of a63r3.

southwest of a64r1 is northeast of a65r1.

southeast of a27r4 is northwest of a66r4.

southeast of a29r0 is northwest of a67r0.

southeast of a31r6 is northwest of a68r6.

southeast of a33r2 is northwest of a69r2.

northwest of a12r1 is southeast of a70r1.

northwest of a14r5 is southeast of a71r5.

southwest of a0r4 is northeast of a72r4.

southwest of a2r0 is northeast of a73r0.

southwest of a4r2 is northeast of a74r2.

southwest of a26r6 is northeast of a75r6.

northeast of a20r8 is southwest of a41r8.

northeast of a17r3 is southwest of a43r3.

northwest of a21r7 is southeast of a45r7.

northwest of a18r8 is southeast of a47r8.

northeast of a7r7 is southwest of a49r7.

north of a16r3 is south of a51r3.

north of a9r3 is south of a35r3.

southwest of a21r7 is northeast of a55r7.

southeast of a7r7 is northwest of a57r7.

southwest of a18r8 is northeast of a59r8.

southeast of a20r8 is northwest of a61r8.

southeast of a17r3 is northwest of a63r3.

northeast of a13r1 is southwest of a65r1.

northeast of a15r5 is southwest of a37r5.

northwest of a8r3 is southeast of a39r3.

east of a5r2 is west of a32r2.

east of a25r6 is west of a30r6.

east of a3r0 is west of a28r0.

east of a1r4 is west of a22r4.

southwest of a8r3 is northeast of a53r3.

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
the player is in a65r1.

the orientation of the player is 6.
