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

[create a23r0]
a23r0 is a area. "An area (23) in r0".
the printed name of the a23r0 is "Room 0".
Understand "Area 23 in Room 0" as a23r0.
the parent of the a23r0 is r0.
the X of the a23r0 is "545890.0086022713".
the Y of the a23r0 is "1699364.0104014394".

[create a32r0]
a32r0 is a area. "An area (32) in r0".
the printed name of the a32r0 is "Room 0".
Understand "Area 32 in Room 0" as a32r0.
the parent of the a32r0 is r0.
the X of the a32r0 is "545786.8816727516".
the Y of the a32r0 is "1699219.4952958585".

[create a34r0]
a34r0 is a area. "An area (34) in r0".
the printed name of the a34r0 is "Room 0".
Understand "Area 34 in Room 0" as a34r0.
the parent of the a34r0 is r0.
the X of the a34r0 is "545903.7496886394".
the Y of the a34r0 is "1699362.317373606".

[create a35r0]
a35r0 is a area. "An area (35) in r0".
the printed name of the a35r0 is "Room 0".
Understand "Area 35 in Room 0" as a35r0.
the parent of the a35r0 is r0.
the X of the a35r0 is "545997.2968536067".
the Y of the a35r0 is "1699251.4209743899".

[create a90r0]
a90r0 is a area. "An area (90) in r0".
the printed name of the a90r0 is "Room 0".
Understand "Area 90 in Room 0" as a90r0.
the parent of the a90r0 is r0.
the X of the a90r0 is "546038.0".
the Y of the a90r0 is "1699200.0".

[create a94r0]
a94r0 is a area. "An area (94) in r0".
the printed name of the a94r0 is "Room 0".
Understand "Area 94 in Room 0" as a94r0.
the parent of the a94r0 is r0.
the X of the a94r0 is "545756.0".
the Y of the a94r0 is "1699172.0".

[create a5r1]
a5r1 is a area. "An area (5) in r1".
the printed name of the a5r1 is "Room 1".
Understand "Area 5 in Room 1" as a5r1.
the parent of the a5r1 is r1.
the X of the a5r1 is "546544.5167685037".
the Y of the a5r1 is "1701173.513395759".

[create a6r1]
a6r1 is a area. "An area (6) in r1".
the printed name of the a6r1 is "Room 1".
Understand "Area 6 in Room 1" as a6r1.
the parent of the a6r1 is r1.
the X of the a6r1 is "546549.8708448356".
the Y of the a6r1 is "1701005.316291368".

[create a11r1]
a11r1 is a area. "An area (11) in r1".
the printed name of the a11r1 is "Room 1".
Understand "Area 11 in Room 1" as a11r1.
the parent of the a11r1 is r1.
the X of the a11r1 is "546477.3333333333".
the Y of the a11r1 is "1701260.9694444444".

[create a50r1]
a50r1 is a area. "An area (50) in r1".
the printed name of the a50r1 is "Room 1".
Understand "Area 50 in Room 1" as a50r1.
the parent of the a50r1 is r1.
the X of the a50r1 is "546644.0".
the Y of the a50r1 is "1701304.0".

[create a51r1]
a51r1 is a area. "An area (51) in r1".
the printed name of the a51r1 is "Room 1".
Understand "Area 51 in Room 1" as a51r1.
the parent of the a51r1 is r1.
the X of the a51r1 is "546610.6666666665".
the Y of the a51r1 is "1701253.5364495001".

[create a93r1]
a93r1 is a area. "An area (93) in r1".
the printed name of the a93r1 is "Room 1".
Understand "Area 93 in Room 1" as a93r1.
the parent of the a93r1 is r1.
the X of the a93r1 is "546444.0".
the Y of the a93r1 is "1701304.0".

[create a24r2]
a24r2 is a area. "An area (24) in r2".
the printed name of the a24r2 is "Room 2".
Understand "Area 24 in Room 2" as a24r2.
the parent of the a24r2 is r2.
the X of the a24r2 is "547144.2390740267".
the Y of the a24r2 is "1699362.431282769".

[create a33r2]
a33r2 is a area. "An area (33) in r2".
the printed name of the a33r2 is "Room 2".
Understand "Area 33 in Room 2" as a33r2.
the parent of the a33r2 is r2.
the X of the a33r2 is "547048.7043713275".
the Y of the a33r2 is "1699222.7067112252".

[create a36r2]
a36r2 is a area. "An area (36) in r2".
the printed name of the a36r2 is "Room 2".
Understand "Area 36 in Room 2" as a36r2.
the parent of the a36r2 is r2.
the X of the a36r2 is "547166.2896516462".
the Y of the a36r2 is "1699360.8833195835".

[create a37r2]
a37r2 is a area. "An area (37) in r2".
the printed name of the a37r2 is "Room 2".
Understand "Area 37 in Room 2" as a37r2.
the parent of the a37r2 is r2.
the X of the a37r2 is "547260.4539705468".
the Y of the a37r2 is "1699248.7643334027".

[create a91r2]
a91r2 is a area. "An area (91) in r2".
the printed name of the a91r2 is "Room 2".
Understand "Area 91 in Room 2" as a91r2.
the parent of the a91r2 is r2.
the X of the a91r2 is "547301.0".
the Y of the a91r2 is "1699200.0".

[create a95r2]
a95r2 is a area. "An area (95) in r2".
the printed name of the a95r2 is "Room 2".
Understand "Area 95 in Room 2" as a95r2.
the parent of the a95r2 is r2.
the X of the a95r2 is "547018.0".
the Y of the a95r2 is "1699172.0".

[create a0r3]
a0r3 is a area. "An area (0) in r3".
the printed name of the a0r3 is "Room 3".
Understand "Area 0 in Room 3" as a0r3.
the parent of the a0r3 is r3.
the X of the a0r3 is "548000.4450183739".
the Y of the a0r3 is "1701135.9641000086".

[create a1r3]
a1r3 is a area. "An area (1) in r3".
the printed name of the a1r3 is "Room 3".
Understand "Area 1 in Room 3" as a1r3.
the parent of the a1r3 is r3.
the X of the a1r3 is "548005.2384354555".
the Y of the a1r3 is "1701140.5902559124".

[create a7r3]
a7r3 is a area. "An area (7) in r3".
the printed name of the a7r3 is "Room 3".
Understand "Area 7 in Room 3" as a7r3.
the parent of the a7r3 is r3.
the X of the a7r3 is "550388.8718497385".
the Y of the a7r3 is "1701195.9455539705".

[create a8r3]
a8r3 is a area. "An area (8) in r3".
the printed name of the a8r3 is "Room 3".
Understand "Area 8 in Room 3" as a8r3.
the parent of the a8r3 is r3.
the X of the a8r3 is "550443.7727272727".
the Y of the a8r3 is "1701281.0".

[create a9r3]
a9r3 is a area. "An area (9) in r3".
the printed name of the a9r3 is "Room 3".
Understand "Area 9 in Room 3" as a9r3.
the parent of the a9r3 is r3.
the X of the a9r3 is "547342.4921500896".
the Y of the a9r3 is "1701260.7986347186".

[create a10r3]
a10r3 is a area. "An area (10) in r3".
the printed name of the a10r3 is "Room 3".
Understand "Area 10 in Room 3" as a10r3.
the parent of the a10r3 is r3.
the X of the a10r3 is "547630.4944995979".
the Y of the a10r3 is "1701211.0205706109".

[create a15r3]
a15r3 is a area. "An area (15) in r3".
the printed name of the a15r3 is "Room 3".
Understand "Area 15 in Room 3" as a15r3.
the parent of the a15r3 is r3.
the X of the a15r3 is "548762.5271327288".
the Y of the a15r3 is "1701191.270065301".

[create a16r3]
a16r3 is a area. "An area (16) in r3".
the printed name of the a16r3 is "Room 3".
Understand "Area 16 in Room 3" as a16r3.
the parent of the a16r3 is r3.
the X of the a16r3 is "547677.1587808633".
the Y of the a16r3 is "1701181.377277184".

[create a19r3]
a19r3 is a area. "An area (19) in r3".
the printed name of the a19r3 is "Room 3".
Understand "Area 19 in Room 3" as a19r3.
the parent of the a19r3 is r3.
the X of the a19r3 is "548838.4256566307".
the Y of the a19r3 is "1700473.7639655606".

[create a20r3]
a20r3 is a area. "An area (20) in r3".
the printed name of the a20r3 is "Room 3".
Understand "Area 20 in Room 3" as a20r3.
the parent of the a20r3 is r3.
the X of the a20r3 is "548920.1312305515".
the Y of the a20r3 is "1700581.7942211537".

[create a21r3]
a21r3 is a area. "An area (21) in r3".
the printed name of the a21r3 is "Room 3".
Understand "Area 21 in Room 3" as a21r3.
the parent of the a21r3 is r3.
the X of the a21r3 is "549004.6176355068".
the Y of the a21r3 is "1700951.0612815882".

[create a22r3]
a22r3 is a area. "An area (22) in r3".
the printed name of the a22r3 is "Room 3".
Understand "Area 22 in Room 3" as a22r3.
the parent of the a22r3 is r3.
the X of the a22r3 is "550462.1150421605".
the Y of the a22r3 is "1700859.7027389589".

[create a38r3]
a38r3 is a area. "An area (38) in r3".
the printed name of the a38r3 is "Room 3".
Understand "Area 38 in Room 3" as a38r3.
the parent of the a38r3 is r3.
the X of the a38r3 is "547589.0".
the Y of the a38r3 is "1701314.0".

[create a39r3]
a39r3 is a area. "An area (39) in r3".
the printed name of the a39r3 is "Room 3".
Understand "Area 39 in Room 3" as a39r3.
the parent of the a39r3 is r3.
the X of the a39r3 is "547620.826460481".
the Y of the a39r3 is "1701305.767353952".

[create a40r3]
a40r3 is a area. "An area (40) in r3".
the printed name of the a40r3 is "Room 3".
Understand "Area 40 in Room 3" as a40r3.
the parent of the a40r3 is r3.
the X of the a40r3 is "548055.0".
the Y of the a40r3 is "1701304.0".

[create a41r3]
a41r3 is a area. "An area (41) in r3".
the printed name of the a41r3 is "Room 3".
Understand "Area 41 in Room 3" as a41r3.
the parent of the a41r3 is r3.
the X of the a41r3 is "548022.005982906".
the Y of the a41r3 is "1701305.7505982905".

[create a42r3]
a42r3 is a area. "An area (42) in r3".
the printed name of the a42r3 is "Room 3".
Understand "Area 42 in Room 3" as a42r3.
the parent of the a42r3 is r3.
the X of the a42r3 is "548805.0".
the Y of the a42r3 is "1701304.0".

[create a43r3]
a43r3 is a area. "An area (43) in r3".
the printed name of the a43r3 is "Room 3".
Understand "Area 43 in Room 3" as a43r3.
the parent of the a43r3 is r3.
the X of the a43r3 is "548771.2".
the Y of the a43r3 is "1701212.7482784986".

[create a44r3]
a44r3 is a area. "An area (44) in r3".
the printed name of the a44r3 is "Room 3".
Understand "Area 44 in Room 3" as a44r3.
the parent of the a44r3 is r3.
the X of the a44r3 is "547286.0".
the Y of the a44r3 is "1701224.0".

[create a45r3]
a45r3 is a area. "An area (45) in r3".
the printed name of the a45r3 is "Room 3".
Understand "Area 45 in Room 3" as a45r3.
the parent of the a45r3 is r3.
the X of the a45r3 is "547326.7622699386".
the Y of the a45r3 is "1701242.6487730062".

[create a54r3]
a54r3 is a area. "An area (54) in r3".
the printed name of the a54r3 is "Room 3".
Understand "Area 54 in Room 3" as a54r3.
the parent of the a54r3 is r3.
the X of the a54r3 is "547295.0".
the Y of the a54r3 is "1701314.0".

[create a55r3]
a55r3 is a area. "An area (55) in r3".
the printed name of the a55r3 is "Room 3".
Understand "Area 55 in Room 3" as a55r3.
the parent of the a55r3 is r3.
the X of the a55r3 is "547325.9153780068".
the Y of the a55r3 is "1701288.1834621993".

[create a66r3]
a66r3 is a area. "An area (66) in r3".
the printed name of the a66r3 is "Room 3".
Understand "Area 66 in Room 3" as a66r3.
the parent of the a66r3 is r3.
the X of the a66r3 is "548620.0".
the Y of the a66r3 is "1700527.0".

[create a67r3]
a67r3 is a area. "An area (67) in r3".
the printed name of the a67r3 is "Room 3".
Understand "Area 67 in Room 3" as a67r3.
the parent of the a67r3 is r3.
the X of the a67r3 is "548768.7814184789".
the Y of the a67r3 is "1700490.4011627906".

[create a68r3]
a68r3 is a area. "An area (68) in r3".
the printed name of the a68r3 is "Room 3".
Understand "Area 68 in Room 3" as a68r3.
the parent of the a68r3 is r3.
the X of the a68r3 is "549162.0".
the Y of the a68r3 is "1700478.0".

[create a69r3]
a69r3 is a area. "An area (69) in r3".
the printed name of the a69r3 is "Room 3".
Understand "Area 69 in Room 3" as a69r3.
the parent of the a69r3 is r3.
the X of the a69r3 is "549073.5780049386".
the Y of the a69r3 is "1700516.8679956896".

[create a70r3]
a70r3 is a area. "An area (70) in r3".
the printed name of the a70r3 is "Room 3".
Understand "Area 70 in Room 3" as a70r3.
the parent of the a70r3 is r3.
the X of the a70r3 is "550430.0".
the Y of the a70r3 is "1700788.0".

[create a71r3]
a71r3 is a area. "An area (71) in r3".
the printed name of the a71r3 is "Room 3".
Understand "Area 71 in Room 3" as a71r3.
the parent of the a71r3 is r3.
the X of the a71r3 is "550438.3098646167".
the Y of the a71r3 is "1700827.908152513".

[create a72r3]
a72r3 is a area. "An area (72) in r3".
the printed name of the a72r3 is "Room 3".
Understand "Area 72 in Room 3" as a72r3.
the parent of the a72r3 is r3.
the X of the a72r3 is "550516.0".
the Y of the a72r3 is "1700825.0".

[create a73r3]
a73r3 is a area. "An area (73) in r3".
the printed name of the a73r3 is "Room 3".
Understand "Area 73 in Room 3" as a73r3.
the parent of the a73r3 is r3.
the X of the a73r3 is "550477.9614546176".
the Y of the a73r3 is "1700854.1909433217".

[create a92r3]
a92r3 is a area. "An area (92) in r3".
the printed name of the a92r3 is "Room 3".
Understand "Area 92 in Room 3" as a92r3.
the parent of the a92r3 is r3.
the X of the a92r3 is "550402.0".
the Y of the a92r3 is "1701286.0".

[create a2r4]
a2r4 is a area. "An area (2) in r4".
the printed name of the a2r4 is "Room 4".
Understand "Area 2 in Room 4" as a2r4.
the parent of the a2r4 is r4.
the X of the a2r4 is "547993.9999999998".
the Y of the a2r4 is "1699368.5778301884".

[create a3r4]
a3r4 is a area. "An area (3) in r4".
the printed name of the a3r4 is "Room 4".
Understand "Area 3 in Room 4" as a3r4.
the parent of the a3r4 is r4.
the X of the a3r4 is "548360.781769306".
the Y of the a3r4 is "1699244.9748289345".

[create a4r4]
a4r4 is a area. "An area (4) in r4".
the printed name of the a4r4 is "Room 4".
Understand "Area 4 in Room 4" as a4r4.
the parent of the a4r4 is r4.
the X of the a4r4 is "547612.389199255".
the Y of the a4r4 is "1699240.2597765364".

[create a17r4]
a17r4 is a area. "An area (17) in r4".
the printed name of the a17r4 is "Room 4".
Understand "Area 17 in Room 4" as a17r4.
the parent of the a17r4 is r4.
the X of the a17r4 is "548000.7860830325".
the Y of the a17r4 is "1700138.5".

[create a18r4]
a18r4 is a area. "An area (18) in r4".
the printed name of the a18r4 is "Room 4".
Understand "Area 18 in Room 4" as a18r4.
the parent of the a18r4 is r4.
the X of the a18r4 is "548770.5803337139".
the Y of the a18r4 is "1700147.1046511626".

[create a25r4]
a25r4 is a area. "An area (25) in r4".
the printed name of the a25r4 is "Room 4".
Understand "Area 25 in Room 4" as a25r4.
the parent of the a25r4 is r4.
the X of the a25r4 is "550410.6979495726".
the Y of the a25r4 is "1699466.6546164062".

[create a26r4]
a26r4 is a area. "An area (26) in r4".
the printed name of the a26r4 is "Room 4".
Understand "Area 26 in Room 4" as a26r4.
the parent of the a26r4 is r4.
the X of the a26r4 is "550408.2063772489".
the Y of the a26r4 is "1699968.2701607419".

[create a27r4]
a27r4 is a area. "An area (27) in r4".
the printed name of the a27r4 is "Room 4".
Understand "Area 27 in Room 4" as a27r4.
the parent of the a27r4 is r4.
the X of the a27r4 is "548800.5244516438".
the Y of the a27r4 is "1699791.7472812526".

[create a28r4]
a28r4 is a area. "An area (28) in r4".
the printed name of the a28r4 is "Room 4".
Understand "Area 28 in Room 4" as a28r4.
the parent of the a28r4 is r4.
the X of the a28r4 is "550054.0200662519".
the Y of the a28r4 is "1700089.2230761896".

[create a29r4]
a29r4 is a area. "An area (29) in r4".
the printed name of the a29r4 is "Room 4".
Understand "Area 29 in Room 4" as a29r4.
the parent of the a29r4 is r4.
the X of the a29r4 is "550517.8477146518".
the Y of the a29r4 is "1700047.3338316958".

[create a30r4]
a30r4 is a area. "An area (30) in r4".
the printed name of the a30r4 is "Room 4".
Understand "Area 30 in Room 4" as a30r4.
the parent of the a30r4 is r4.
the X of the a30r4 is "550688.8618197276".
the Y of the a30r4 is "1700094.1585884353".

[create a31r4]
a31r4 is a area. "An area (31) in r4".
the printed name of the a31r4 is "Room 4".
Understand "Area 31 in Room 4" as a31r4.
the parent of the a31r4 is r4.
the X of the a31r4 is "549263.3005528575".
the Y of the a31r4 is "1699298.8583478576".

[create a58r4]
a58r4 is a area. "An area (58) in r4".
the printed name of the a58r4 is "Room 4".
Understand "Area 58 in Room 4" as a58r4.
the parent of the a58r4 is r4.
the X of the a58r4 is "549997.0".
the Y of the a58r4 is "1700054.0".

[create a59r4]
a59r4 is a area. "An area (59) in r4".
the printed name of the a59r4 is "Room 4".
Understand "Area 59 in Room 4" as a59r4.
the parent of the a59r4 is r4.
the X of the a59r4 is "550039.2240682533".
the Y of the a59r4 is "1700072.207678491".

[create a60r4]
a60r4 is a area. "An area (60) in r4".
the printed name of the a60r4 is "Room 4".
Understand "Area 60 in Room 4" as a60r4.
the parent of the a60r4 is r4.
the X of the a60r4 is "550697.0".
the Y of the a60r4 is "1700045.0".

[create a61r4]
a61r4 is a area. "An area (61) in r4".
the printed name of the a61r4 is "Room 4".
Understand "Area 61 in Room 4" as a61r4.
the parent of the a61r4 is r4.
the X of the a61r4 is "550715.2704191617".
the Y of the a61r4 is "1700073.6734131735".

[create a62r4]
a62r4 is a area. "An area (62) in r4".
the printed name of the a62r4 is "Room 4".
Understand "Area 62 in Room 4" as a62r4.
the parent of the a62r4 is r4.
the X of the a62r4 is "550006.0".
the Y of the a62r4 is "1700143.0".

[create a63r4]
a63r4 is a area. "An area (63) in r4".
the printed name of the a63r4 is "Room 4".
Understand "Area 63 in Room 4" as a63r4.
the parent of the a63r4 is r4.
the X of the a63r4 is "550037.0420577894".
the Y of the a63r4 is "1700117.3833874145".

[create a64r4]
a64r4 is a area. "An area (64) in r4".
the printed name of the a64r4 is "Room 4".
Understand "Area 64 in Room 4" as a64r4.
the parent of the a64r4 is r4.
the X of the a64r4 is "550734.0".
the Y of the a64r4 is "1700143.0".

[create a65r4]
a65r4 is a area. "An area (65) in r4".
the printed name of the a65r4 is "Room 4".
Understand "Area 65 in Room 4" as a65r4.
the parent of the a65r4 is r4.
the X of the a65r4 is "550702.7862838914".
the Y of the a65r4 is "1700117.3660287082".

[create a74r4]
a74r4 is a area. "An area (74) in r4".
the printed name of the a74r4 is "Room 4".
Understand "Area 74 in Room 4" as a74r4.
the parent of the a74r4 is r4.
the X of the a74r4 is "547567.0".
the Y of the a74r4 is "1699191.0".

[create a75r4]
a75r4 is a area. "An area (75) in r4".
the printed name of the a75r4 is "Room 4".
Understand "Area 75 in Room 4" as a75r4.
the parent of the a75r4 is r4.
the X of the a75r4 is "547598.2137161084".
the Y of the a75r4 is "1699216.633971292".

[create a76r4]
a76r4 is a area. "An area (76) in r4".
the printed name of the a76r4 is "Room 4".
Understand "Area 76 in Room 4" as a76r4.
the parent of the a76r4 is r4.
the X of the a76r4 is "547558.0".
the Y of the a76r4 is "1699280.0".

[create a77r4]
a77r4 is a area. "An area (77) in r4".
the printed name of the a77r4 is "Room 4".
Understand "Area 77 in Room 4" as a77r4.
the parent of the a77r4 is r4.
the X of the a77r4 is "547589.0059757691".
the Y of the a77r4 is "1699260.657907662".

[create a78r4]
a78r4 is a area. "An area (78) in r4".
the printed name of the a78r4 is "Room 4".
Understand "Area 78 in Room 4" as a78r4.
the parent of the a78r4 is r4.
the X of the a78r4 is "548409.0".
the Y of the a78r4 is "1699191.0".

[create a79r4]
a79r4 is a area. "An area (79) in r4".
the printed name of the a79r4 is "Room 4".
Understand "Area 79 in Room 4" as a79r4.
the parent of the a79r4 is r4.
the X of the a79r4 is "548377.7862838916".
the Y of the a79r4 is "1699216.633971292".

[create a80r4]
a80r4 is a area. "An area (80) in r4".
the printed name of the a80r4 is "Room 4".
Understand "Area 80 in Room 4" as a80r4.
the parent of the a80r4 is r4.
the X of the a80r4 is "548418.0".
the Y of the a80r4 is "1699280.0".

[create a81r4]
a81r4 is a area. "An area (81) in r4".
the printed name of the a81r4 is "Room 4".
Understand "Area 81 in Room 4" as a81r4.
the parent of the a81r4 is r4.
the X of the a81r4 is "548375.1485252036".
the Y of the a81r4 is "1699261.8557671143".

[create a82r4]
a82r4 is a area. "An area (82) in r4".
the printed name of the a82r4 is "Room 4".
Understand "Area 82 in Room 4" as a82r4.
the parent of the a82r4 is r4.
the X of the a82r4 is "549210.0".
the Y of the a82r4 is "1699163.0".

[create a83r4]
a83r4 is a area. "An area (83) in r4".
the printed name of the a83r4 is "Room 4".
Understand "Area 83 in Room 4" as a83r4.
the parent of the a83r4 is r4.
the X of the a83r4 is "549246.0".
the Y of the a83r4 is "1699255.3867768594".

[create a84r4]
a84r4 is a area. "An area (84) in r4".
the printed name of the a84r4 is "Room 4".
Understand "Area 84 in Room 4" as a84r4.
the parent of the a84r4 is r4.
the X of the a84r4 is "550553.0".
the Y of the a84r4 is "1699369.0".

[create a85r4]
a85r4 is a area. "An area (85) in r4".
the printed name of the a85r4 is "Room 4".
Understand "Area 85 in Room 4" as a85r4.
the parent of the a85r4 is r4.
the X of the a85r4 is "550494.3376066436".
the Y of the a85r4 is "1699410.7846478492".

[create a86r4]
a86r4 is a area. "An area (86) in r4".
the printed name of the a86r4 is "Room 4".
Understand "Area 86 in Room 4" as a86r4.
the parent of the a86r4 is r4.
the X of the a86r4 is "548620.0".
the Y of the a86r4 is "1699713.0".

[create a87r4]
a87r4 is a area. "An area (87) in r4".
the printed name of the a87r4 is "Room 4".
Understand "Area 87 in Room 4" as a87r4.
the parent of the a87r4 is r4.
the X of the a87r4 is "548702.470724191".
the Y of the a87r4 is "1699747.7".

[create a88r4]
a88r4 is a area. "An area (88) in r4".
the printed name of the a88r4 is "Room 4".
Understand "Area 88 in Room 4" as a88r4.
the parent of the a88r4 is r4.
the X of the a88r4 is "550583.0".
the Y of the a88r4 is "1699986.0".

[create a89r4]
a89r4 is a area. "An area (89) in r4".
the printed name of the a89r4 is "Room 4".
Understand "Area 89 in Room 4" as a89r4.
the parent of the a89r4 is r4.
the X of the a89r4 is "550520.1666666667".
the Y of the a89r4 is "1700045.8333333333".

[create a12r5]
a12r5 is a area. "An area (12) in r5".
the printed name of the a12r5 is "Room 5".
Understand "Area 12 in Room 5" as a12r5.
the parent of the a12r5 is r5.
the X of the a12r5 is "545050.2813718208".
the Y of the a12r5 is "1701261.2845894434".

[create a13r5]
a13r5 is a area. "An area (13) in r5".
the printed name of the a13r5 is "Room 5".
Understand "Area 13 in Room 5" as a13r5.
the parent of the a13r5 is r5.
the X of the a13r5 is "545405.79770735".
the Y of the a13r5 is "1701144.5677680378".

[create a14r5]
a14r5 is a area. "An area (14) in r5".
the printed name of the a14r5 is "Room 5".
Understand "Area 14 in Room 5" as a14r5.
the parent of the a14r5 is r5.
the X of the a14r5 is "545779.1653756485".
the Y of the a14r5 is "1701261.2658517451".

[create a46r5]
a46r5 is a area. "An area (46) in r5".
the printed name of the a46r5 is "Room 5".
Understand "Area 46 in Room 5" as a46r5.
the parent of the a46r5 is r5.
the X of the a46r5 is "544994.0".
the Y of the a46r5 is "1701224.0".

[create a47r5]
a47r5 is a area. "An area (47) in r5".
the printed name of the a47r5 is "Room 5".
Understand "Area 47 in Room 5" as a47r5.
the parent of the a47r5 is r5.
the X of the a47r5 is "545033.4941364606".
the Y of the a47r5 is "1701242.7755863538".

[create a48r5]
a48r5 is a area. "An area (48) in r5".
the printed name of the a48r5 is "Room 5".
Understand "Area 48 in Room 5" as a48r5.
the parent of the a48r5 is r5.
the X of the a48r5 is "545836.0".
the Y of the a48r5 is "1701224.0".

[create a49r5]
a49r5 is a area. "An area (49) in r5".
the printed name of the a49r5 is "Room 5".
Understand "Area 49 in Room 5" as a49r5.
the parent of the a49r5 is r5.
the X of the a49r5 is "545795.6686930091".
the Y of the a49r5 is "1701242.2965214453".

[create a52r5]
a52r5 is a area. "An area (52) in r5".
the printed name of the a52r5 is "Room 5".
Understand "Area 52 in Room 5" as a52r5.
the parent of the a52r5 is r5.
the X of the a52r5 is "545826.0".
the Y of the a52r5 is "1701314.0".

[create a53r5]
a53r5 is a area. "An area (53) in r5".
the printed name of the a53r5 is "Room 5".
Understand "Area 53 in Room 5" as a53r5.
the parent of the a53r5 is r5.
the X of the a53r5 is "545795.1050372066".
the Y of the a53r5 is "1701287.7894485786".

[create a56r5]
a56r5 is a area. "An area (56) in r5".
the printed name of the a56r5 is "Room 5".
Understand "Area 56 in Room 5" as a56r5.
the parent of the a56r5 is r5.
the X of the a56r5 is "545003.0".
the Y of the a56r5 is "1701314.0".

[create a57r5]
a57r5 is a area. "An area (57) in r5".
the printed name of the a57r5 is "Room 5".
Understand "Area 57 in Room 5" as a57r5.
the parent of the a57r5 is r5.
the X of the a57r5 is "545034.0876068375".
the Y of the a57r5 is "1701288.166239316".

[create d0]
d0 is a door. "A door between a36r2 and a16r3".
d0 is north of a36r2 and south of a16r3.

[create d2]
d2 is a door. "A door between a34r0 and a6r1".
d2 is north of a34r0 and south of a6r1.

[create d4]
d4 is a door. "A door between a23r0 and a13r5".
d4 is north of a23r0 and south of a13r5.

[create d6]
d6 is a door. "A door between a6r1 and a24r2".
d6 is southeast of a6r1 and northwest of a24r2.

[create d8]
d8 is a door. "A door between a0r3 and a17r4".
d8 is south of a0r3 and north of a17r4.

[create d9]
d9 is a door. "A door between a19r3 and a18r4".
d9 is south of a19r3 and north of a18r4.

northeast of a0r3 is southwest of a1r3.

east of a2r4 is west of a3r4.

west of a2r4 is east of a4r4.

south of a5r1 is north of a6r1.

northeast of a7r3 is southwest of a8r3.

east of a9r3 is west of a10r3.

northwest of a5r1 is southeast of a11r1.

east of a12r5 is west of a13r5.

east of a13r5 is west of a14r5.

east of a1r3 is west of a15r3.

west of a0r3 is east of a16r3.

east of a17r4 is west of a18r4.

northeast of a19r3 is southwest of a20r3.

north of a20r3 is south of a21r3.

south of a7r3 is north of a22r3.

west of a7r3 is east of a21r3.

southeast of a15r3 is northwest of a21r3.

north of a2r4 is south of a17r4.

north of a25r4 is south of a26r4.

south of a18r4 is north of a27r4.

west of a26r4 is east of a28r4.

northeast of a26r4 is southwest of a29r4.

east of a29r4 is west of a30r4.

west of a25r4 is east of a31r4.

southwest of a23r0 is northeast of a32r0.

southwest of a24r2 is northeast of a33r2.

southeast of a27r4 is northwest of a31r4.

southeast of a34r0 is northwest of a35r0.

southeast of a36r2 is northwest of a37r2.

east of a38r3 is west of a39r3.

west of a40r3 is east of a41r3.

south of a42r3 is north of a43r3.

northeast of a44r3 is southwest of a45r3.

northeast of a46r5 is southwest of a47r5.

northwest of a48r5 is southeast of a49r5.

southwest of a50r1 is northeast of a51r1.

southwest of a52r5 is northeast of a53r5.

southeast of a54r3 is northwest of a55r3.

southeast of a56r5 is northwest of a57r5.

northeast of a58r4 is southwest of a59r4.

northeast of a60r4 is southwest of a61r4.

southeast of a62r4 is northwest of a63r4.

southwest of a64r4 is northeast of a65r4.

east of a66r3 is west of a67r3.

northwest of a68r3 is southeast of a69r3.

north of a70r3 is south of a71r3.

northwest of a72r3 is southeast of a73r3.

northeast of a74r4 is southwest of a75r4.

southeast of a76r4 is northwest of a77r4.

northwest of a78r4 is southeast of a79r4.

southwest of a80r4 is northeast of a81r4.

north of a82r4 is south of a83r4.

northwest of a84r4 is southeast of a85r4.

northeast of a86r4 is southwest of a87r4.

northwest of a88r4 is southeast of a89r4.

southeast of a35r0 is northwest of a90r0.

southeast of a37r2 is northwest of a91r2.

west of a8r3 is east of a92r3.

northwest of a11r1 is southeast of a93r1.

southwest of a32r0 is northeast of a94r0.

southwest of a33r2 is northeast of a95r2.

north of a1r3 is south of a41r3.

southwest of a9r3 is northeast of a45r3.

southwest of a12r5 is northeast of a47r5.

southeast of a14r5 is northwest of a49r5.

northeast of a5r1 is southwest of a51r1.

northeast of a14r5 is southwest of a53r5.

northwest of a9r3 is southeast of a55r3.

northwest of a12r5 is southeast of a57r5.

north of a10r3 is south of a39r3.

northwest of a28r4 is southeast of a63r4.

northeast of a30r4 is southwest of a65r4.

west of a19r3 is east of a67r3.

southeast of a20r3 is northwest of a69r3.

southwest of a22r3 is northeast of a71r3.

east of a22r3 is west of a73r3.

north of a15r3 is south of a43r3.

southeast of a10r3 is northwest of a16r3.

south of a31r4 is north of a83r4.

east of a24r2 is west of a36r2.

east of a23r0 is west of a34r0.

southeast of a25r4 is northwest of a85r4.

southwest of a27r4 is northeast of a87r4.

southeast of a29r4 is northwest of a89r4.

southwest of a28r4 is northeast of a59r4.

southeast of a30r4 is northwest of a61r4.

southwest of a4r4 is northeast of a75r4.

northwest of a4r4 is southeast of a77r4.

southeast of a3r4 is northwest of a79r4.

northeast of a3r4 is southwest of a81r4.

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
the player is in a13r5.

the orientation of the player is 6.
