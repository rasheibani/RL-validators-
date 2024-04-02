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

[create a25r0]
a25r0 is a area. "An area (25) in r0".
the printed name of the a25r0 is "Room 0".
Understand "Area 25 in Room 0" as a25r0.
the parent of the a25r0 is r0.
the X of the a25r0 is "548822.7496886395".
the Y of the a25r0 is "1763962.317373606".

[create a26r0]
a26r0 is a area. "An area (26) in r0".
the printed name of the a26r0 is "Room 0".
Understand "Area 26 in Room 0" as a26r0.
the parent of the a26r0 is r0.
the X of the a26r0 is "548916.2968536066".
the Y of the a26r0 is "1763851.4209743903".

[create a31r0]
a31r0 is a area. "An area (31) in r0".
the printed name of the a31r0 is "Room 0".
Understand "Area 31 in Room 0" as a31r0.
the parent of the a31r0 is r0.
the X of the a31r0 is "548808.9861188938".
the Y of the a31r0 is "1763964.0131715974".

[create a39r0]
a39r0 is a area. "An area (39) in r0".
the printed name of the a39r0 is "Room 0".
Understand "Area 39 in Room 0" as a39r0.
the parent of the a39r0 is r0.
the X of the a39r0 is "548705.8224123432".
the Y of the a39r0 is "1763820.0921328291".

[create a107r0]
a107r0 is a area. "An area (107) in r0".
the printed name of the a107r0 is "Room 0".
Understand "Area 107 in Room 0" as a107r0.
the parent of the a107r0 is r0.
the X of the a107r0 is "548675.0".
the Y of the a107r0 is "1763772.0".

[create a109r0]
a109r0 is a area. "An area (109) in r0".
the printed name of the a109r0 is "Room 0".
Understand "Area 109 in Room 0" as a109r0.
the parent of the a109r0 is r0.
the X of the a109r0 is "548957.0".
the Y of the a109r0 is "1763800.0".

[create a7r1]
a7r1 is a area. "An area (7) in r1".
the printed name of the a7r1 is "Room 1".
Understand "Area 7 in Room 1" as a7r1.
the parent of the a7r1 is r1.
the X of the a7r1 is "549396.333333333".
the Y of the a7r1 is "1765860.9694444444".

[create a8r1]
a8r1 is a area. "An area (8) in r1".
the printed name of the a8r1 is "Room 1".
Understand "Area 8 in Room 1" as a8r1.
the parent of the a8r1 is r1.
the X of the a8r1 is "549463.5167685037".
the Y of the a8r1 is "1765773.5133957588".

[create a23r1]
a23r1 is a area. "An area (23) in r1".
the printed name of the a23r1 is "Room 1".
Understand "Area 23 in Room 1" as a23r1.
the parent of the a23r1 is r1.
the X of the a23r1 is "549468.8708448355".
the Y of the a23r1 is "1765605.3162913683".

[create a65r1]
a65r1 is a area. "An area (65) in r1".
the printed name of the a65r1 is "Room 1".
Understand "Area 65 in Room 1" as a65r1.
the parent of the a65r1 is r1.
the X of the a65r1 is "549563.0".
the Y of the a65r1 is "1765904.0".

[create a66r1]
a66r1 is a area. "An area (66) in r1".
the printed name of the a66r1 is "Room 1".
Understand "Area 66 in Room 1" as a66r1.
the parent of the a66r1 is r1.
the X of the a66r1 is "549529.6666666667".
the Y of the a66r1 is "1765853.5364495006".

[create a111r1]
a111r1 is a area. "An area (111) in r1".
the printed name of the a111r1 is "Room 1".
Understand "Area 111 in Room 1" as a111r1.
the parent of the a111r1 is r1.
the X of the a111r1 is "549363.0".
the Y of the a111r1 is "1765904.0".

[create a27r2]
a27r2 is a area. "An area (27) in r2".
the printed name of the a27r2 is "Room 2".
Understand "Area 27 in Room 2" as a27r2.
the parent of the a27r2 is r2.
the X of the a27r2 is "550085.2896516463".
the Y of the a27r2 is "1763960.8833195833".

[create a28r2]
a28r2 is a area. "An area (28) in r2".
the printed name of the a28r2 is "Room 2".
Understand "Area 28 in Room 2" as a28r2.
the parent of the a28r2 is r2.
the X of the a28r2 is "550179.4539705468".
the Y of the a28r2 is "1763848.7643334027".

[create a32r2]
a32r2 is a area. "An area (32) in r2".
the printed name of the a32r2 is "Room 2".
Understand "Area 32 in Room 2" as a32r2.
the parent of the a32r2 is r2.
the X of the a32r2 is "550063.2390740267".
the Y of the a32r2 is "1763962.4312827692".

[create a40r2]
a40r2 is a area. "An area (40) in r2".
the printed name of the a40r2 is "Room 2".
Understand "Area 40 in Room 2" as a40r2.
the parent of the a40r2 is r2.
the X of the a40r2 is "549967.7043713275".
the Y of the a40r2 is "1763822.706711225".

[create a108r2]
a108r2 is a area. "An area (108) in r2".
the printed name of the a108r2 is "Room 2".
Understand "Area 108 in Room 2" as a108r2.
the parent of the a108r2 is r2.
the X of the a108r2 is "549937.0".
the Y of the a108r2 is "1763772.0".

[create a110r2]
a110r2 is a area. "An area (110) in r2".
the printed name of the a110r2 is "Room 2".
Understand "Area 110 in Room 2" as a110r2.
the parent of the a110r2 is r2.
the X of the a110r2 is "550220.0".
the Y of the a110r2 is "1763800.0".

[create a15r3]
a15r3 is a area. "An area (15) in r3".
the printed name of the a15r3 is "Room 3".
Understand "Area 15 in Room 3" as a15r3.
the parent of the a15r3 is r3.
the X of the a15r3 is "550261.4921500895".
the Y of the a15r3 is "1765860.7986347186".

[create a16r3]
a16r3 is a area. "An area (16) in r3".
the printed name of the a16r3 is "Room 3".
Understand "Area 16 in Room 3" as a16r3.
the parent of the a16r3 is r3.
the X of the a16r3 is "550595.9486065779".
the Y of the a16r3 is "1765778.1064398696".

[create a18r3]
a18r3 is a area. "An area (18) in r3".
the printed name of the a18r3 is "Room 3".
Understand "Area 18 in Room 3" as a18r3.
the parent of the a18r3 is r3.
the X of the a18r3 is "550925.5509290383".
the Y of the a18r3 is "1765861.0062088708".

[create a43r3]
a43r3 is a area. "An area (43) in r3".
the printed name of the a43r3 is "Room 3".
Understand "Area 43 in Room 3" as a43r3.
the parent of the a43r3 is r3.
the X of the a43r3 is "550214.0".
the Y of the a43r3 is "1765914.0".

[create a44r3]
a44r3 is a area. "An area (44) in r3".
the printed name of the a44r3 is "Room 3".
Understand "Area 44 in Room 3" as a44r3.
the parent of the a44r3 is r3.
the X of the a44r3 is "550244.9153780071".
the Y of the a44r3 is "1765888.1834621993".

[create a49r3]
a49r3 is a area. "An area (49) in r3".
the printed name of the a49r3 is "Room 3".
Understand "Area 49 in Room 3" as a49r3.
the parent of the a49r3 is r3.
the X of the a49r3 is "550973.0".
the Y of the a49r3 is "1765914.0".

[create a50r3]
a50r3 is a area. "An area (50) in r3".
the printed name of the a50r3 is "Room 3".
Understand "Area 50 in Room 3" as a50r3.
the parent of the a50r3 is r3.
the X of the a50r3 is "550941.9123931625".
the Y of the a50r3 is "1765888.1662393163".

[create a81r3]
a81r3 is a area. "An area (81) in r3".
the printed name of the a81r3 is "Room 3".
Understand "Area 81 in Room 3" as a81r3.
the parent of the a81r3 is r3.
the X of the a81r3 is "550205.0".
the Y of the a81r3 is "1765824.0".

[create a82r3]
a82r3 is a area. "An area (82) in r3".
the printed name of the a82r3 is "Room 3".
Understand "Area 82 in Room 3" as a82r3.
the parent of the a82r3 is r3.
the X of the a82r3 is "550245.7622699386".
the Y of the a82r3 is "1765842.6487730062".

[create a83r3]
a83r3 is a area. "An area (83) in r3".
the printed name of the a83r3 is "Room 3".
Understand "Area 83 in Room 3" as a83r3.
the parent of the a83r3 is r3.
the X of the a83r3 is "550982.0".
the Y of the a83r3 is "1765824.0".

[create a84r3]
a84r3 is a area. "An area (84) in r3".
the printed name of the a84r3 is "Room 3".
Understand "Area 84 in Room 3" as a84r3.
the parent of the a84r3 is r3.
the X of the a84r3 is "550941.662577747".
the Y of the a84r3 is "1765842.6912577746".

[create a0r4]
a0r4 is a area. "An area (0) in r4".
the printed name of the a0r4 is "Room 4".
Understand "Area 0 in Room 4" as a0r4.
the parent of the a0r4 is r4.
the X of the a0r4 is "546531.3656942605".
the Y of the a0r4 is "1765833.1395782027".

[create a1r4]
a1r4 is a area. "An area (1) in r4".
the printed name of the a1r4 is "Room 4".
Understand "Area 1 in Room 4" as a1r4.
the parent of the a1r4 is r4.
the X of the a1r4 is "546613.6099758961".
the Y of the a1r4 is "1765774.8205626053".

[create a2r4]
a2r4 is a area. "An area (2) in r4".
the printed name of the a2r4 is "Room 4".
Understand "Area 2 in Room 4" as a2r4.
the parent of the a2r4 is r4.
the X of the a2r4 is "546269.7072666287".
the Y of the a2r4 is "1765861.4777749956".

[create a3r4]
a3r4 is a area. "An area (3) in r4".
the printed name of the a3r4 is "Room 4".
Understand "Area 3 in Room 4" as a3r4.
the parent of the a3r4 is r4.
the X of the a3r4 is "546936.2338247439".
the Y of the a3r4 is "1765778.194453643".

[create a4r4]
a4r4 is a area. "An area (4) in r4".
the printed name of the a4r4 is "Room 4".
Understand "Area 4 in Room 4" as a4r4.
the parent of the a4r4 is r4.
the X of the a4r4 is "547286.9343690763".
the Y of the a4r4 is "1765861.6427193342".

[create a9r4]
a9r4 is a area. "An area (9) in r4".
the printed name of the a9r4 is "Room 4".
Understand "Area 9 in Room 4" as a9r4.
the parent of the a9r4 is r4.
the X of the a9r4 is "547761.9589252131".
the Y of the a9r4 is "1765861.6795222145".

[create a10r4]
a10r4 is a area. "An area (10) in r4".
the printed name of the a10r4 is "Room 4".
Understand "Area 10 in Room 4" as a10r4.
the parent of the a10r4 is r4.
the X of the a10r4 is "547960.3701298703".
the Y of the a10r4 is "1765842.945887446".

[create a13r4]
a13r4 is a area. "An area (13) in r4".
the printed name of the a13r4 is "Room 4".
Understand "Area 13 in Room 4" as a13r4.
the parent of the a13r4 is r4.
the X of the a13r4 is "548319.3787096874".
the Y of the a13r4 is "1765743.7282370713".

[create a14r4]
a14r4 is a area. "An area (14) in r4".
the printed name of the a14r4 is "Room 4".
Understand "Area 14 in Room 4" as a14r4.
the parent of the a14r4 is r4.
the X of the a14r4 is "548432.8869813074".
the Y of the a14r4 is "1765815.9284325424".

[create a17r4]
a17r4 is a area. "An area (17) in r4".
the printed name of the a17r4 is "Room 4".
Understand "Area 17 in Room 4" as a17r4.
the parent of the a17r4 is r4.
the X of the a17r4 is "548114.3619459793".
the Y of the a17r4 is "1765765.4949057794".

[create a19r4]
a19r4 is a area. "An area (19) in r4".
the printed name of the a19r4 is "Room 4".
Understand "Area 19 in Room 4" as a19r4.
the parent of the a19r4 is r4.
the X of the a19r4 is "548698.1653756481".
the Y of the a19r4 is "1765861.2658517456".

[create a21r4]
a21r4 is a area. "An area (21) in r4".
the printed name of the a21r4 is "Room 4".
Understand "Area 21 in Room 4" as a21r4.
the parent of the a21r4 is r4.
the X of the a21r4 is "547483.6892610957".
the Y of the a21r4 is "1764910.9362679885".

[create a22r4]
a22r4 is a area. "An area (22) in r4".
the printed name of the a22r4 is "Room 4".
Understand "Area 22 in Room 4" as a22r4.
the parent of the a22r4 is r4.
the X of the a22r4 is "546904.9420276638".
the Y of the a22r4 is "1765743.5929472565".

[create a29r4]
a29r4 is a area. "An area (29) in r4".
the printed name of the a29r4 is "Room 4".
Understand "Area 29 in Room 4" as a29r4.
the parent of the a29r4 is r4.
the X of the a29r4 is "547496.0873366152".
the Y of the a29r4 is "1764849.159650314".

[create a45r4]
a45r4 is a area. "An area (45) in r4".
the printed name of the a45r4 is "Room 4".
Understand "Area 45 in Room 4" as a45r4.
the parent of the a45r4 is r4.
the X of the a45r4 is "547334.0".
the Y of the a45r4 is "1765914.0".

[create a46r4]
a46r4 is a area. "An area (46) in r4".
the printed name of the a46r4 is "Room 4".
Understand "Area 46 in Room 4" as a46r4.
the parent of the a46r4 is r4.
the X of the a46r4 is "547302.9123931625".
the Y of the a46r4 is "1765888.166239316".

[create a51r4]
a51r4 is a area. "An area (51) in r4".
the printed name of the a51r4 is "Room 4".
Understand "Area 51 in Room 4" as a51r4.
the parent of the a51r4 is r4.
the X of the a51r4 is "546982.1111111111".
the Y of the a51r4 is "1765904.0".

[create a52r4]
a52r4 is a area. "An area (52) in r4".
the printed name of the a52r4 is "Room 4".
Understand "Area 52 in Room 4" as a52r4.
the parent of the a52r4 is r4.
the X of the a52r4 is "546949.1923615087".
the Y of the a52r4 is "1765905.4040895503".

[create a53r4]
a53r4 is a area. "An area (53) in r4".
the printed name of the a53r4 is "Room 4".
Understand "Area 53 in Room 4" as a53r4.
the parent of the a53r4 is r4.
the X of the a53r4 is "546491.0".
the Y of the a53r4 is "1765904.0".

[create a54r4]
a54r4 is a area. "An area (54) in r4".
the printed name of the a54r4 is "Room 4".
Understand "Area 54 in Room 4" as a54r4.
the parent of the a54r4 is r4.
the X of the a54r4 is "546523.9940170939".
the Y of the a54r4 is "1765905.7505982905".

[create a55r4]
a55r4 is a area. "An area (55) in r4".
the printed name of the a55r4 is "Room 4".
Understand "Area 55 in Room 4" as a55r4.
the parent of the a55r4 is r4.
the X of the a55r4 is "547921.0".
the Y of the a55r4 is "1765904.0".

[create a56r4]
a56r4 is a area. "An area (56) in r4".
the printed name of the a56r4 is "Room 4".
Understand "Area 56 in Room 4" as a56r4.
the parent of the a56r4 is r4.
the X of the a56r4 is "547953.9940170939".
the Y of the a56r4 is "1765905.7505982905".

[create a57r4]
a57r4 is a area. "An area (57) in r4".
the printed name of the a57r4 is "Room 4".
Understand "Area 57 in Room 4" as a57r4.
the parent of the a57r4 is r4.
the X of the a57r4 is "548475.0".
the Y of the a57r4 is "1765904.0".

[create a58r4]
a58r4 is a area. "An area (58) in r4".
the printed name of the a58r4 is "Room 4".
Understand "Area 58 in Room 4" as a58r4.
the parent of the a58r4 is r4.
the X of the a58r4 is "548442.005982906".
the Y of the a58r4 is "1765905.7505982905".

[create a61r4]
a61r4 is a area. "An area (61) in r4".
the printed name of the a61r4 is "Room 4".
Understand "Area 61 in Room 4" as a61r4.
the parent of the a61r4 is r4.
the X of the a61r4 is "546213.0".
the Y of the a61r4 is "1765824.0".

[create a62r4]
a62r4 is a area. "An area (62) in r4".
the printed name of the a62r4 is "Room 4".
Understand "Area 62 in Room 4" as a62r4.
the parent of the a62r4 is r4.
the X of the a62r4 is "546251.7029286582".
the Y of the a62r4 is "1765842.477452371".

[create a63r4]
a63r4 is a area. "An area (63) in r4".
the printed name of the a63r4 is "Room 4".
Understand "Area 63 in Room 4" as a63r4.
the parent of the a63r4 is r4.
the X of the a63r4 is "547343.0".
the Y of the a63r4 is "1765824.0".

[create a64r4]
a64r4 is a area. "An area (64) in r4".
the printed name of the a64r4 is "Room 4".
Understand "Area 64 in Room 4" as a64r4.
the parent of the a64r4 is r4.
the X of the a64r4 is "547304.0502336449".
the Y of the a64r4 is "1765842.8300233646".

[create a67r4]
a67r4 is a area. "An area (67) in r4".
the printed name of the a67r4 is "Room 4".
Understand "Area 67 in Room 4" as a67r4.
the parent of the a67r4 is r4.
the X of the a67r4 is "548745.0".
the Y of the a67r4 is "1765914.0".

[create a68r4]
a68r4 is a area. "An area (68) in r4".
the printed name of the a68r4 is "Room 4".
Understand "Area 68 in Room 4" as a68r4.
the parent of the a68r4 is r4.
the X of the a68r4 is "548714.1050372066".
the Y of the a68r4 is "1765887.7894485784".

[create a69r4]
a69r4 is a area. "An area (69) in r4".
the printed name of the a69r4 is "Room 4".
Understand "Area 69 in Room 4" as a69r4.
the parent of the a69r4 is r4.
the X of the a69r4 is "546223.0".
the Y of the a69r4 is "1765914.0".

[create a70r4]
a70r4 is a area. "An area (70) in r4".
the printed name of the a70r4 is "Room 4".
Understand "Area 70 in Room 4" as a70r4.
the parent of the a70r4 is r4.
the X of the a70r4 is "546253.8949627934".
the Y of the a70r4 is "1765887.7894485786".

[create a71r4]
a71r4 is a area. "An area (71) in r4".
the printed name of the a71r4 is "Room 4".
Understand "Area 71 in Room 4" as a71r4.
the parent of the a71r4 is r4.
the X of the a71r4 is "547715.0".
the Y of the a71r4 is "1765914.0".

[create a72r4]
a72r4 is a area. "An area (72) in r4".
the printed name of the a72r4 is "Room 4".
Understand "Area 72 in Room 4" as a72r4.
the parent of the a72r4 is r4.
the X of the a72r4 is "547745.9153780066".
the Y of the a72r4 is "1765888.1834621993".

[create a87r4]
a87r4 is a area. "An area (87) in r4".
the printed name of the a87r4 is "Room 4".
Understand "Area 87 in Room 4" as a87r4.
the parent of the a87r4 is r4.
the X of the a87r4 is "547706.0".
the Y of the a87r4 is "1765824.0".

[create a88r4]
a88r4 is a area. "An area (88) in r4".
the printed name of the a88r4 is "Room 4".
Understand "Area 88 in Room 4" as a88r4.
the parent of the a88r4 is r4.
the X of the a88r4 is "547744.6344476745".
the Y of the a88r4 is "1765842.8615552324".

[create a89r4]
a89r4 is a area. "An area (89) in r4".
the printed name of the a89r4 is "Room 4".
Understand "Area 89 in Room 4" as a89r4.
the parent of the a89r4 is r4.
the X of the a89r4 is "548755.0".
the Y of the a89r4 is "1765824.0".

[create a90r4]
a90r4 is a area. "An area (90) in r4".
the printed name of the a90r4 is "Room 4".
Understand "Area 90 in Room 4" as a90r4.
the parent of the a90r4 is r4.
the X of the a90r4 is "548714.6686930091".
the Y of the a90r4 is "1765842.2965214455".

[create a30r5]
a30r5 is a area. "An area (30) in r5".
the printed name of the a30r5 is "Room 5".
Understand "Area 30 in Room 5" as a30r5.
the parent of the a30r5 is r5.
the X of the a30r5 is "548116.3355180039".
the Y of the a30r5 is "1763962.3355180037".

[create a34r5]
a34r5 is a area. "An area (34) in r5".
the printed name of the a34r5 is "Room 5".
Understand "Area 34 in Room 5" as a34r5.
the parent of the a34r5 is r5.
the X of the a34r5 is "547738.0670611439".
the Y of the a34r5 is "1763844.7228796843".

[create a35r5]
a35r5 is a area. "An area (35) in r5".
the printed name of the a35r5 is "Room 5".
Understand "Area 35 in Room 5" as a35r5.
the parent of the a35r5 is r5.
the X of the a35r5 is "546797.8633422253".
the Y of the a35r5 is "1763934.3385130367".

[create a36r5]
a36r5 is a area. "An area (36) in r5".
the printed name of the a36r5 is "Room 5".
Understand "Area 36 in Room 5" as a36r5.
the parent of the a36r5 is r5.
the X of the a36r5 is "547146.0171235863".
the Y of the a36r5 is "1763842.9159051348".

[create a37r5]
a37r5 is a area. "An area (37) in r5".
the printed name of the a37r5 is "Room 5".
Understand "Area 37 in Room 5" as a37r5.
the parent of the a37r5 is r5.
the X of the a37r5 is "548485.7426126124".
the Y of the a37r5 is "1763841.7067567569".

[create a38r5]
a38r5 is a area. "An area (38) in r5".
the printed name of the a38r5 is "Room 5".
Understand "Area 38 in Room 5" as a38r5.
the parent of the a38r5 is r5.
the X of the a38r5 is "546480.0941579641".
the Y of the a38r5 is "1763841.5825458067".

[create a75r5]
a75r5 is a area. "An area (75) in r5".
the printed name of the a75r5 is "Room 5".
Understand "Area 75 in Room 5" as a75r5.
the parent of the a75r5 is r5.
the X of the a75r5 is "547202.0".
the Y of the a75r5 is "1763880.0".

[create a76r5]
a76r5 is a area. "An area (76) in r5".
the printed name of the a76r5 is "Room 5".
Understand "Area 76 in Room 5" as a76r5.
the parent of the a76r5 is r5.
the X of the a76r5 is "547163.279721321".
the Y of the a76r5 is "1763861.4380057093".

[create a77r5]
a77r5 is a area. "An area (77) in r5".
the printed name of the a77r5 is "Room 5".
Understand "Area 77 in Room 5" as a77r5.
the parent of the a77r5 is r5.
the X of the a77r5 is "547681.0".
the Y of the a77r5 is "1763880.0".

[create a78r5]
a78r5 is a area. "An area (78) in r5".
the printed name of the a78r5 is "Room 5".
Understand "Area 78 in Room 5" as a78r5.
the parent of the a78r5 is r5.
the X of the a78r5 is "547723.2240682533".
the Y of the a78r5 is "1763861.792321509".

[create a91r5]
a91r5 is a area. "An area (91) in r5".
the printed name of the a91r5 is "Room 5".
Understand "Area 91 in Room 5" as a91r5.
the parent of the a91r5 is r5.
the X of the a91r5 is "546434.0".
the Y of the a91r5 is "1763791.0".

[create a92r5]
a92r5 is a area. "An area (92) in r5".
the printed name of the a92r5 is "Room 5".
Understand "Area 92 in Room 5" as a92r5.
the parent of the a92r5 is r5.
the X of the a92r5 is "546465.0420577896".
the Y of the a92r5 is "1763816.6166125853".

[create a93r5]
a93r5 is a area. "An area (93) in r5".
the printed name of the a93r5 is "Room 5".
Understand "Area 93 in Room 5" as a93r5.
the parent of the a93r5 is r5.
the X of the a93r5 is "548532.0".
the Y of the a93r5 is "1763791.0".

[create a94r5]
a94r5 is a area. "An area (94) in r5".
the printed name of the a94r5 is "Room 5".
Understand "Area 94 in Room 5" as a94r5.
the parent of the a94r5 is r5.
the X of the a94r5 is "548500.7862838916".
the Y of the a94r5 is "1763816.6339712918".

[create a95r5]
a95r5 is a area. "An area (95) in r5".
the printed name of the a95r5 is "Room 5".
Understand "Area 95 in Room 5" as a95r5.
the parent of the a95r5 is r5.
the X of the a95r5 is "547193.0".
the Y of the a95r5 is "1763791.0".

[create a96r5]
a96r5 is a area. "An area (96) in r5".
the printed name of the a96r5 is "Room 5".
Understand "Area 96 in Room 5" as a96r5.
the parent of the a96r5 is r5.
the X of the a96r5 is "547161.7862838916".
the Y of the a96r5 is "1763816.6339712918".

[create a99r5]
a99r5 is a area. "An area (99) in r5".
the printed name of the a99r5 is "Room 5".
Understand "Area 99 in Room 5" as a99r5.
the parent of the a99r5 is r5.
the X of the a99r5 is "547690.0".
the Y of the a99r5 is "1763791.0".

[create a100r5]
a100r5 is a area. "An area (100) in r5".
the printed name of the a100r5 is "Room 5".
Understand "Area 100 in Room 5" as a100r5.
the parent of the a100r5 is r5.
the X of the a100r5 is "547721.2137161084".
the Y of the a100r5 is "1763816.6339712918".

[create a103r5]
a103r5 is a area. "An area (103) in r5".
the printed name of the a103r5 is "Room 5".
Understand "Area 103 in Room 5" as a103r5.
the parent of the a103r5 is r5.
the X of the a103r5 is "546425.0".
the Y of the a103r5 is "1763880.0".

[create a104r5]
a104r5 is a area. "An area (104) in r5".
the printed name of the a104r5 is "Room 5".
Understand "Area 104 in Room 5" as a104r5.
the parent of the a104r5 is r5.
the X of the a104r5 is "546461.018362578".
the Y of the a104r5 is "1763861.1647782384".

[create a105r5]
a105r5 is a area. "An area (105) in r5".
the printed name of the a105r5 is "Room 5".
Understand "Area 105 in Room 5" as a105r5.
the parent of the a105r5 is r5.
the X of the a105r5 is "548541.0".
the Y of the a105r5 is "1763880.0".

[create a106r5]
a106r5 is a area. "An area (106) in r5".
the printed name of the a106r5 is "Room 5".
Understand "Area 106 in Room 5" as a106r5.
the parent of the a106r5 is r5.
the X of the a106r5 is "548504.1452295459".
the Y of the a106r5 is "1763861.24935881".

[create a5r6]
a5r6 is a area. "An area (5) in r6".
the printed name of the a5r6 is "Room 6".
Understand "Area 5 in Room 6" as a5r6.
the parent of the a5r6 is r6.
the X of the a5r6 is "545983.0".
the Y of the a5r6 is "1763968.577830189".

[create a6r6]
a6r6 is a area. "An area (6) in r6".
the printed name of the a6r6 is "Room 6".
Understand "Area 6 in Room 6" as a6r6.
the parent of the a6r6 is r6.
the X of the a6r6 is "546349.7290448346".
the Y of the a6r6 is "1763845.0627030537".

[create a20r6]
a20r6 is a area. "An area (20) in r6".
the printed name of the a20r6 is "Room 6".
Understand "Area 20 in Room 6" as a20r6.
the parent of the a20r6 is r6.
the X of the a20r6 is "545978.1730804583".
the Y of the a20r6 is "1764659.7005024818".

[create a33r6]
a33r6 is a area. "An area (33) in r6".
the printed name of the a33r6 is "Room 6".
Understand "Area 33 in Room 6" as a33r6.
the parent of the a33r6 is r6.
the X of the a33r6 is "545601.3945333334".
the Y of the a33r6 is "1763840.2686666667".

[create a41r6]
a41r6 is a area. "An area (41) in r6".
the printed name of the a41r6 is "Room 6".
Understand "Area 41 in Room 6" as a41r6.
the parent of the a41r6 is r6.
the X of the a41r6 is "545556.0".
the Y of the a41r6 is "1763791.0".

[create a42r6]
a42r6 is a area. "An area (42) in r6".
the printed name of the a42r6 is "Room 6".
Understand "Area 42 in Room 6" as a42r6.
the parent of the a42r6 is r6.
the X of the a42r6 is "545587.2137161085".
the Y of the a42r6 is "1763816.633971292".

[create a79r6]
a79r6 is a area. "An area (79) in r6".
the printed name of the a79r6 is "Room 6".
Understand "Area 79 in Room 6" as a79r6.
the parent of the a79r6 is r6.
the X of the a79r6 is "546407.0".
the Y of the a79r6 is "1763880.0".

[create a80r6]
a80r6 is a area. "An area (80) in r6".
the printed name of the a80r6 is "Room 6".
Understand "Area 80 in Room 6" as a80r6.
the parent of the a80r6 is r6.
the X of the a80r6 is "546363.9383292026".
the Y of the a80r6 is "1763861.87702289".

[create a97r6]
a97r6 is a area. "An area (97) in r6".
the printed name of the a97r6 is "Room 6".
Understand "Area 97 in Room 6" as a97r6.
the parent of the a97r6 is r6.
the X of the a97r6 is "545547.0".
the Y of the a97r6 is "1763880.0".

[create a98r6]
a98r6 is a area. "An area (98) in r6".
the printed name of the a98r6 is "Room 6".
Understand "Area 98 in Room 6" as a98r6.
the parent of the a98r6 is r6.
the X of the a98r6 is "545577.8730024244".
the Y of the a98r6 is "1763860.6444609193".

[create a101r6]
a101r6 is a area. "An area (101) in r6".
the printed name of the a101r6 is "Room 6".
Understand "Area 101 in Room 6" as a101r6.
the parent of the a101r6 is r6.
the X of the a101r6 is "546398.0".
the Y of the a101r6 is "1763791.0".

[create a102r6]
a102r6 is a area. "An area (102) in r6".
the printed name of the a102r6 is "Room 6".
Understand "Area 102 in Room 6" as a102r6.
the parent of the a102r6 is r6.
the X of the a102r6 is "546366.7862838916".
the Y of the a102r6 is "1763816.633971292".

[create a11r7]
a11r7 is a area. "An area (11) in r7".
the printed name of the a11r7 is "Room 7".
Understand "Area 11 in Room 7" as a11r7.
the parent of the a11r7 is r7.
the X of the a11r7 is "545044.0175949462".
the Y of the a11r7 is "1765861.5825998152".

[create a12r7]
a12r7 is a area. "An area (12) in r7".
the printed name of the a12r7 is "Room 7".
Understand "Area 12 in Room 7" as a12r7.
the parent of the a12r7 is r7.
the X of the a12r7 is "545409.1530110595".
the Y of the a12r7 is "1765737.6714275307".

[create a24r7]
a24r7 is a area. "An area (24) in r7".
the printed name of the a24r7 is "Room 7".
Understand "Area 24 in Room 7" as a24r7.
the parent of the a24r7 is r7.
the X of the a24r7 is "545791.9151692868".
the Y of the a24r7 is "1765861.6108476832".

[create a47r7]
a47r7 is a area. "An area (47) in r7".
the printed name of the a47r7 is "Room 7".
Understand "Area 47 in Room 7" as a47r7.
the parent of the a47r7 is r7.
the X of the a47r7 is "545839.0".
the Y of the a47r7 is "1765914.0".

[create a48r7]
a48r7 is a area. "An area (48) in r7".
the printed name of the a48r7 is "Room 7".
Understand "Area 48 in Room 7" as a48r7.
the parent of the a48r7 is r7.
the X of the a48r7 is "545807.9123931623".
the Y of the a48r7 is "1765888.166239316".

[create a59r7]
a59r7 is a area. "An area (59) in r7".
the printed name of the a59r7 is "Room 7".
Understand "Area 59 in Room 7" as a59r7.
the parent of the a59r7 is r7.
the X of the a59r7 is "545848.0".
the Y of the a59r7 is "1765824.0".

[create a60r7]
a60r7 is a area. "An area (60) in r7".
the printed name of the a60r7 is "Room 7".
Understand "Area 60 in Room 7" as a60r7.
the parent of the a60r7 is r7.
the X of the a60r7 is "545808.9448424068".
the Y of the a60r7 is "1765842.8194842406".

[create a73r7]
a73r7 is a area. "An area (73) in r7".
the printed name of the a73r7 is "Room 7".
Understand "Area 73 in Room 7" as a73r7.
the parent of the a73r7 is r7.
the X of the a73r7 is "544997.0".
the Y of the a73r7 is "1765914.0".

[create a74r7]
a74r7 is a area. "An area (74) in r7".
the printed name of the a74r7 is "Room 7".
Understand "Area 74 in Room 7" as a74r7.
the parent of the a74r7 is r7.
the X of the a74r7 is "545027.9153780068".
the Y of the a74r7 is "1765888.1834621995".

[create a85r7]
a85r7 is a area. "An area (85) in r7".
the printed name of the a85r7 is "Room 7".
Understand "Area 85 in Room 7" as a85r7.
the parent of the a85r7 is r7.
the X of the a85r7 is "544988.0".
the Y of the a85r7 is "1765824.0".

[create a86r7]
a86r7 is a area. "An area (86) in r7".
the printed name of the a86r7 is "Room 7".
Understand "Area 86 in Room 7" as a86r7.
the parent of the a86r7 is r7.
the X of the a86r7 is "545026.2466592428".
the Y of the a86r7 is "1765842.9003340756".

[create d0]
d0 is a door. "A door between a29r4 and a30r5".
d0 is southeast of a29r4 and northwest of a30r5.

[create d1]
d1 is a door. "A door between a29r4 and a35r5".
d1 is southwest of a29r4 and northeast of a35r5.

[create d4]
d4 is a door. "A door between a27r2 and a16r3".
d4 is north of a27r2 and south of a16r3.

[create d6]
d6 is a door. "A door between a25r0 and a23r1".
d6 is north of a25r0 and south of a23r1.

[create d8]
d8 is a door. "A door between a31r0 and a13r4".
d8 is north of a31r0 and south of a13r4.

[create d10]
d10 is a door. "A door between a23r1 and a32r2".
d10 is southeast of a23r1 and northwest of a32r2.

[create d12]
d12 is a door. "A door between a1r4 and a20r6".
d12 is southwest of a1r4 and northeast of a20r6.

[create d14]
d14 is a door. "A door between a20r6 and a12r7".
d14 is northwest of a20r6 and southeast of a12r7.

southeast of a0r4 is northwest of a1r4.

west of a0r4 is east of a2r4.

east of a3r4 is west of a4r4.

east of a5r6 is west of a6r6.

southeast of a7r1 is northwest of a8r1.

east of a9r4 is west of a10r4.

east of a11r7 is west of a12r7.

northeast of a13r4 is southwest of a14r4.

east of a15r3 is west of a16r3.

southeast of a10r4 is northwest of a17r4.

east of a16r3 is west of a18r3.

east of a14r4 is west of a19r4.

southwest of a17r4 is northeast of a21r4.

northwest of a21r4 is southeast of a22r4.

south of a8r1 is north of a23r1.

east of a12r7 is west of a24r7.

west of a13r4 is east of a17r4.

east of a1r4 is west of a22r4.

southeast of a25r0 is northwest of a26r0.

southeast of a27r2 is northwest of a28r2.

north of a5r6 is south of a20r6.

west of a5r6 is east of a33r6.

west of a30r5 is east of a34r5.

east of a35r5 is west of a36r5.

east of a30r5 is west of a37r5.

west of a35r5 is east of a38r5.

southwest of a31r0 is northeast of a39r0.

southwest of a32r2 is northeast of a40r2.

northeast of a41r6 is southwest of a42r6.

southeast of a43r3 is northwest of a44r3.

southwest of a45r4 is northeast of a46r4.

southwest of a47r7 is northeast of a48r7.

southwest of a49r3 is northeast of a50r3.

west of a51r4 is east of a52r4.

east of a53r4 is west of a54r4.

east of a55r4 is west of a56r4.

west of a57r4 is east of a58r4.

northwest of a59r7 is southeast of a60r7.

northeast of a61r4 is southwest of a62r4.

northwest of a63r4 is southeast of a64r4.

southwest of a65r1 is northeast of a66r1.

southwest of a67r4 is northeast of a68r4.

southeast of a69r4 is northwest of a70r4.

southeast of a71r4 is northwest of a72r4.

southeast of a73r7 is northwest of a74r7.

southwest of a75r5 is northeast of a76r5.

southeast of a77r5 is northwest of a78r5.

southwest of a79r6 is northeast of a80r6.

northeast of a81r3 is southwest of a82r3.

northwest of a83r3 is southeast of a84r3.

northeast of a85r7 is southwest of a86r7.

northeast of a87r4 is southwest of a88r4.

northwest of a89r4 is southeast of a90r4.

northeast of a91r5 is southwest of a92r5.

northwest of a93r5 is southeast of a94r5.

northwest of a95r5 is southeast of a96r5.

southeast of a97r6 is northwest of a98r6.

northeast of a99r5 is southwest of a100r5.

northwest of a101r6 is southeast of a102r6.

southeast of a103r5 is northwest of a104r5.

southwest of a105r5 is northeast of a106r5.

southwest of a39r0 is northeast of a107r0.

southwest of a40r2 is northeast of a108r2.

southeast of a26r0 is northwest of a109r0.

southeast of a28r2 is northwest of a110r2.

northwest of a7r1 is southeast of a111r1.

southwest of a33r6 is northeast of a42r6.

southwest of a38r5 is northeast of a92r5.

southeast of a37r5 is northwest of a94r5.

southeast of a36r5 is northwest of a96r5.

northwest of a33r6 is southeast of a98r6.

southwest of a34r5 is northeast of a100r5.

southeast of a6r6 is northwest of a102r6.

northeast of a4r4 is southwest of a46r4.

northeast of a24r7 is southwest of a48r7.

northeast of a18r3 is southwest of a50r3.

north of a3r4 is south of a52r4.

north of a0r4 is south of a54r4.

north of a10r4 is south of a56r4.

north of a14r4 is south of a58r4.

southwest of a2r4 is northeast of a62r4.

southeast of a4r4 is northwest of a64r4.

northeast of a8r1 is southwest of a66r1.

northeast of a19r4 is southwest of a68r4.

northwest of a2r4 is southeast of a70r4.

northwest of a9r4 is southeast of a72r4.

northwest of a11r7 is southeast of a74r7.

northwest of a15r3 is southeast of a44r3.

south of a21r4 is north of a29r4.

southwest of a3r4 is northeast of a22r4.

southwest of a15r3 is northeast of a82r3.

southeast of a18r3 is northwest of a84r3.

southwest of a11r7 is northeast of a86r7.

southwest of a9r4 is northeast of a88r4.

southeast of a19r4 is northwest of a90r4.

southeast of a24r7 is northwest of a60r7.

northwest of a38r5 is southeast of a104r5.

northeast of a37r5 is southwest of a106r5.

northeast of a36r5 is southwest of a76r5.

northwest of a34r5 is southeast of a78r5.

northeast of a6r6 is southwest of a80r6.

west of a27r2 is east of a32r2.

west of a25r0 is east of a31r0.

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
the player is in a49r3.

the orientation of the player is 6.
