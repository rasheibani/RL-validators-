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

[create a20r0]
a20r0 is a area. "An area (20) in r0".
the printed name of the a20r0 is "Room 0".
Understand "Area 20 in Room 0" as a20r0.
the parent of the a20r0 is r0.
the X of the a20r0 is "545545.6602166065".
the Y of the a20r0 is "1365815.0".

[create a21r0]
a21r0 is a area. "An area (21) in r0".
the printed name of the a21r0 is "Room 0".
Understand "Area 21 in Room 0" as a21r0.
the parent of the a21r0 is r0.
the X of the a21r0 is "545525.0".
the Y of the a21r0 is "1364968.5778301884".

[create a32r0]
a32r0 is a area. "An area (32) in r0".
the printed name of the a32r0 is "Room 0".
Understand "Area 32 in Room 0" as a32r0.
the parent of the a32r0 is r0.
the X of the a32r0 is "545891.7817693062".
the Y of the a32r0 is "1364844.9748289345".

[create a34r0]
a34r0 is a area. "An area (34) in r0".
the printed name of the a34r0 is "Room 0".
Understand "Area 34 in Room 0" as a34r0.
the parent of the a34r0 is r0.
the X of the a34r0 is "545143.389199255".
the Y of the a34r0 is "1364840.2597765361".

[create a80r0]
a80r0 is a area. "An area (80) in r0".
the printed name of the a80r0 is "Room 0".
Understand "Area 80 in Room 0" as a80r0.
the parent of the a80r0 is r0.
the X of the a80r0 is "545089.0".
the Y of the a80r0 is "1364880.0".

[create a81r0]
a81r0 is a area. "An area (81) in r0".
the printed name of the a81r0 is "Room 0".
Understand "Area 81 in Room 0" as a81r0.
the parent of the a81r0 is r0.
the X of the a81r0 is "545120.0059757694".
the Y of the a81r0 is "1364860.657907662".

[create a82r0]
a82r0 is a area. "An area (82) in r0".
the printed name of the a82r0 is "Room 0".
Understand "Area 82 in Room 0" as a82r0.
the parent of the a82r0 is r0.
the X of the a82r0 is "545940.0".
the Y of the a82r0 is "1364791.0".

[create a83r0]
a83r0 is a area. "An area (83) in r0".
the printed name of the a83r0 is "Room 0".
Understand "Area 83 in Room 0" as a83r0.
the parent of the a83r0 is r0.
the X of the a83r0 is "545908.7862838916".
the Y of the a83r0 is "1364816.6339712918".

[create a86r0]
a86r0 is a area. "An area (86) in r0".
the printed name of the a86r0 is "Room 0".
Understand "Area 86 in Room 0" as a86r0.
the parent of the a86r0 is r0.
the X of the a86r0 is "545949.0".
the Y of the a86r0 is "1364880.0".

[create a87r0]
a87r0 is a area. "An area (87) in r0".
the printed name of the a87r0 is "Room 0".
Understand "Area 87 in Room 0" as a87r0.
the parent of the a87r0 is r0.
the X of the a87r0 is "545906.1485252037".
the Y of the a87r0 is "1364861.855767114".

[create a95r0]
a95r0 is a area. "An area (95) in r0".
the printed name of the a95r0 is "Room 0".
Understand "Area 95 in Room 0" as a95r0.
the parent of the a95r0 is r0.
the X of the a95r0 is "545098.0".
the Y of the a95r0 is "1364791.0".

[create a96r0]
a96r0 is a area. "An area (96) in r0".
the printed name of the a96r0 is "Room 0".
Understand "Area 96 in Room 0" as a96r0.
the parent of the a96r0 is r0.
the X of the a96r0 is "545129.2137161085".
the Y of the a96r0 is "1364816.633971292".

[create a0r1]
a0r1 is a area. "An area (0) in r1".
the printed name of the a0r1 is "Room 1".
Understand "Area 0 in Room 1" as a0r1.
the parent of the a0r1 is r1.
the X of the a0r1 is "546741.7012552448".
the Y of the a0r1 is "1366833.4362519374".

[create a1r1]
a1r1 is a area. "An area (1) in r1".
the printed name of the a1r1 is "Room 1".
Understand "Area 1 in Room 1" as a1r1.
the parent of the a1r1 is r1.
the X of the a1r1 is "546978.799560993".
the Y of the a1r1 is "1366735.1405519834".

[create a4r1]
a4r1 is a area. "An area (4) in r1".
the printed name of the a4r1 is "Room 1".
Understand "Area 4 in Room 1" as a4r1.
the parent of the a4r1 is r1.
the X of the a4r1 is "546615.3690201745".
the Y of the a4r1 is "1366783.6762775735".

[create a5r1]
a5r1 is a area. "An area (5) in r1".
the printed name of the a5r1 is "Room 1".
Understand "Area 5 in Room 1" as a5r1.
the parent of the a5r1 is r1.
the X of the a5r1 is "547340.9392056433".
the Y of the a5r1 is "1366850.3160699778".

[create a6r1]
a6r1 is a area. "An area (6) in r1".
the printed name of the a6r1 is "Room 1".
Understand "Area 6 in Room 1" as a6r1.
the parent of the a6r1 is r1.
the X of the a6r1 is "547483.0571718668".
the Y of the a6r1 is "1366863.6890744825".

[create a9r1]
a9r1 is a area. "An area (9) in r1".
the printed name of the a9r1 is "Room 1".
Understand "Area 9 in Room 1" as a9r1.
the parent of the a9r1 is r1.
the X of the a9r1 is "545168.092671901".
the Y of the a9r1 is "1366859.8065726862".

[create a10r1]
a10r1 is a area. "An area (10) in r1".
the printed name of the a10r1 is "Room 1".
Understand "Area 10 in Room 1" as a10r1.
the parent of the a10r1 is r1.
the X of the a10r1 is "545525.0".
the Y of the a10r1 is "1366735.3372641508".

[create a11r1]
a11r1 is a area. "An area (11) in r1".
the printed name of the a11r1 is "Room 1".
Understand "Area 11 in Room 1" as a11r1.
the parent of the a11r1 is r1.
the X of the a11r1 is "545882.8606168448".
the Y of the a11r1 is "1366859.8602906289".

[create a12r1]
a12r1 is a area. "An area (12) in r1".
the printed name of the a12r1 is "Room 1".
Understand "Area 12 in Room 1" as a12r1.
the parent of the a12r1 is r1.
the X of the a12r1 is "547042.6643320168".
the Y of the a12r1 is "1366672.913307503".

[create a14r1]
a14r1 is a area. "An area (14) in r1".
the printed name of the a14r1 is "Room 1".
Understand "Area 14 in Room 1" as a14r1.
the parent of the a14r1 is r1.
the X of the a14r1 is "546317.4139571367".
the Y of the a14r1 is "1366859.2758094773".

[create a15r1]
a15r1 is a area. "An area (15) in r1".
the printed name of the a15r1 is "Room 1".
Understand "Area 15 in Room 1" as a15r1.
the parent of the a15r1 is r1.
the X of the a15r1 is "546611.4900277876".
the Y of the a15r1 is "1366788.0776018454".

[create a18r1]
a18r1 is a area. "An area (18) in r1".
the printed name of the a18r1 is "Room 1".
Understand "Area 18 in Room 1" as a18r1.
the parent of the a18r1 is r1.
the X of the a18r1 is "545913.7347933757".
the Y of the a18r1 is "1365969.6178680474".

[create a22r1]
a22r1 is a area. "An area (22) in r1".
the printed name of the a22r1 is "Room 1".
Understand "Area 22 in Room 1" as a22r1.
the parent of the a22r1 is r1.
the X of the a22r1 is "547088.7200883401".
the Y of the a22r1 is "1366376.1599239258".

[create a23r1]
a23r1 is a area. "An area (23) in r1".
the printed name of the a23r1 is "Room 1".
Understand "Area 23 in Room 1" as a23r1.
the parent of the a23r1 is r1.
the X of the a23r1 is "547260.4857855833".
the Y of the a23r1 is "1366120.2263084948".

[create a38r1]
a38r1 is a area. "An area (38) in r1".
the printed name of the a38r1 is "Room 1".
Understand "Area 38 in Room 1" as a38r1.
the parent of the a38r1 is r1.
the X of the a38r1 is "547532.0".
the Y of the a38r1 is "1366914.0".

[create a39r1]
a39r1 is a area. "An area (39) in r1".
the printed name of the a39r1 is "Room 1".
Understand "Area 39 in Room 1" as a39r1.
the parent of the a39r1 is r1.
the X of the a39r1 is "547494.7409469251".
the Y of the a39r1 is "1366887.082327436".

[create a40r1]
a40r1 is a area. "An area (40) in r1".
the printed name of the a40r1 is "Room 1".
Understand "Area 40 in Room 1" as a40r1.
the parent of the a40r1 is r1.
the X of the a40r1 is "547029.0".
the Y of the a40r1 is "1366904.0".

[create a41r1]
a41r1 is a area. "An area (41) in r1".
the printed name of the a41r1 is "Room 1".
Understand "Area 41 in Room 1" as a41r1.
the parent of the a41r1 is r1.
the X of the a41r1 is "547001.7416953037".
the Y of the a41r1 is "1366906.3241695303".

[create a42r1]
a42r1 is a area. "An area (42) in r1".
the printed name of the a42r1 is "Room 1".
Understand "Area 42 in Room 1" as a42r1.
the parent of the a42r1 is r1.
the X of the a42r1 is "546709.0".
the Y of the a42r1 is "1366914.0".

[create a43r1]
a43r1 is a area. "An area (43) in r1".
the printed name of the a43r1 is "Room 1".
Understand "Area 43 in Room 1" as a43r1.
the parent of the a43r1 is r1.
the X of the a43r1 is "546735.3263855847".
the Y of the a43r1 is "1366907.6449387614".

[create a44r1]
a44r1 is a area. "An area (44) in r1".
the printed name of the a44r1 is "Room 1".
Understand "Area 44 in Room 1" as a44r1.
the parent of the a44r1 is r1.
the X of the a44r1 is "547377.7989690722".
the Y of the a44r1 is "1366906.0103092783".

[create a45r1]
a45r1 is a area. "An area (45) in r1".
the printed name of the a45r1 is "Room 1".
Understand "Area 45 in Room 1" as a45r1.
the parent of the a45r1 is r1.
the X of the a45r1 is "547344.9539210503".
the Y of the a45r1 is "1366906.7605982905".

[create a46r1]
a46r1 is a area. "An area (46) in r1".
the printed name of the a46r1 is "Room 1".
Understand "Area 46 in Room 1" as a46r1.
the parent of the a46r1 is r1.
the X of the a46r1 is "546565.0".
the Y of the a46r1 is "1366904.0".

[create a47r1]
a47r1 is a area. "An area (47) in r1".
the printed name of the a47r1 is "Room 1".
Understand "Area 47 in Room 1" as a47r1.
the parent of the a47r1 is r1.
the X of the a47r1 is "546601.2170727496".
the Y of the a47r1 is "1366905.428292725".

[create a52r1]
a52r1 is a area. "An area (52) in r1".
the printed name of the a52r1 is "Room 1".
Understand "Area 52 in Room 1" as a52r1.
the parent of the a52r1 is r1.
the X of the a52r1 is "547542.0".
the Y of the a52r1 is "1366824.0".

[create a53r1]
a53r1 is a area. "An area (53) in r1".
the printed name of the a53r1 is "Room 1".
Understand "Area 53 in Room 1" as a53r1.
the parent of the a53r1 is r1.
the X of the a53r1 is "547501.6686930091".
the Y of the a53r1 is "1366842.2965214453".

[create a54r1]
a54r1 is a area. "An area (54) in r1".
the printed name of the a54r1 is "Room 1".
Understand "Area 54 in Room 1" as a54r1.
the parent of the a54r1 is r1.
the X of the a54r1 is "545120.0".
the Y of the a54r1 is "1366914.0".

[create a55r1]
a55r1 is a area. "An area (55) in r1".
the printed name of the a55r1 is "Room 1".
Understand "Area 55 in Room 1" as a55r1.
the parent of the a55r1 is r1.
the X of the a55r1 is "545150.9153780069".
the Y of the a55r1 is "1366888.1834621993".

[create a56r1]
a56r1 is a area. "An area (56) in r1".
the printed name of the a56r1 is "Room 1".
Understand "Area 56 in Room 1" as a56r1.
the parent of the a56r1 is r1.
the X of the a56r1 is "546269.0".
the Y of the a56r1 is "1366914.0".

[create a57r1]
a57r1 is a area. "An area (57) in r1".
the printed name of the a57r1 is "Room 1".
Understand "Area 57 in Room 1" as a57r1.
the parent of the a57r1 is r1.
the X of the a57r1 is "546299.9153780069".
the Y of the a57r1 is "1366888.1834621993".

[create a60r1]
a60r1 is a area. "An area (60) in r1".
the printed name of the a60r1 is "Room 1".
Understand "Area 60 in Room 1" as a60r1.
the parent of the a60r1 is r1.
the X of the a60r1 is "545931.0".
the Y of the a60r1 is "1366914.0".

[create a61r1]
a61r1 is a area. "An area (61) in r1".
the printed name of the a61r1 is "Room 1".
Understand "Area 61 in Room 1" as a61r1.
the parent of the a61r1 is r1.
the X of the a61r1 is "545899.9123931625".
the Y of the a61r1 is "1366888.1662393163".

[create a70r1]
a70r1 is a area. "An area (70) in r1".
the printed name of the a70r1 is "Room 1".
Understand "Area 70 in Room 1" as a70r1.
the parent of the a70r1 is r1.
the X of the a70r1 is "547110.0".
the Y of the a70r1 is "1366120.9820359282".

[create a71r1]
a71r1 is a area. "An area (71) in r1".
the printed name of the a71r1 is "Room 1".
Understand "Area 71 in Room 1" as a71r1.
the parent of the a71r1 is r1.
the X of the a71r1 is "547181.0039880322".
the Y of the a71r1 is "1366099.486526946".

[create a72r1]
a72r1 is a area. "An area (72) in r1".
the printed name of the a72r1 is "Room 1".
Understand "Area 72 in Room 1" as a72r1.
the parent of the a72r1 is r1.
the X of the a72r1 is "546260.0".
the Y of the a72r1 is "1366824.0".

[create a73r1]
a73r1 is a area. "An area (73) in r1".
the printed name of the a73r1 is "Room 1".
Understand "Area 73 in Room 1" as a73r1.
the parent of the a73r1 is r1.
the X of the a73r1 is "546303.1182354635".
the Y of the a73r1 is "1366842.4131764537".

[create a74r1]
a74r1 is a area. "An area (74) in r1".
the printed name of the a74r1 is "Room 1".
Understand "Area 74 in Room 1" as a74r1.
the parent of the a74r1 is r1.
the X of the a74r1 is "545111.0".
the Y of the a74r1 is "1366824.0".

[create a75r1]
a75r1 is a area. "An area (75) in r1".
the printed name of the a75r1 is "Room 1".
Understand "Area 75 in Room 1" as a75r1.
the parent of the a75r1 is r1.
the X of the a75r1 is "545153.06".
the Y of the a75r1 is "1366842.5189999999".

[create a76r1]
a76r1 is a area. "An area (76) in r1".
the printed name of the a76r1 is "Room 1".
Understand "Area 76 in Room 1" as a76r1.
the parent of the a76r1 is r1.
the X of the a76r1 is "545940.0".
the Y of the a76r1 is "1366824.0".

[create a77r1]
a77r1 is a area. "An area (77) in r1".
the printed name of the a77r1 is "Room 1".
Understand "Area 77 in Room 1" as a77r1.
the parent of the a77r1 is r1.
the X of the a77r1 is "545897.94".
the Y of the a77r1 is "1366842.519".

[create a26r2]
a26r2 is a area. "An area (26) in r2".
the printed name of the a26r2 is "Room 2".
Understand "Area 26 in Room 2" as a26r2.
the parent of the a26r2 is r2.
the X of the a26r2 is "547609.7496886395".
the Y of the a26r2 is "1364962.317373606".

[create a27r2]
a27r2 is a area. "An area (27) in r2".
the printed name of the a27r2 is "Room 2".
Understand "Area 27 in Room 2" as a27r2.
the parent of the a27r2 is r2.
the X of the a27r2 is "547703.2968536067".
the Y of the a27r2 is "1364851.4209743894".

[create a31r2]
a31r2 is a area. "An area (31) in r2".
the printed name of the a31r2 is "Room 2".
Understand "Area 31 in Room 2" as a31r2.
the parent of the a31r2 is r2.
the X of the a31r2 is "547595.9876667359".
the Y of the a31r2 is "1364964.0129808893".

[create a35r2]
a35r2 is a area. "An area (35) in r2".
the printed name of the a35r2 is "Room 2".
Understand "Area 35 in Room 2" as a35r2.
the parent of the a35r2 is r2.
the X of the a35r2 is "547492.8249575008".
the Y of the a35r2 is "1364820.0664994568".

[create a36r2]
a36r2 is a area. "An area (36) in r2".
the printed name of the a36r2 is "Room 2".
Understand "Area 36 in Room 2" as a36r2.
the parent of the a36r2 is r2.
the X of the a36r2 is "547561.2475565866".
the Y of the a36r2 is "1364917.9667456239".

[create a67r2]
a67r2 is a area. "An area (67) in r2".
the printed name of the a67r2 is "Room 2".
Understand "Area 67 in Room 2" as a67r2.
the parent of the a67r2 is r2.
the X of the a67r2 is "547433.4583392278".
the Y of the a67r2 is "1364884.6217467196".

[create a97r2]
a97r2 is a area. "An area (97) in r2".
the printed name of the a97r2 is "Room 2".
Understand "Area 97 in Room 2" as a97r2.
the parent of the a97r2 is r2.
the X of the a97r2 is "547462.0".
the Y of the a97r2 is "1364772.0".

[create a99r2]
a99r2 is a area. "An area (99) in r2".
the printed name of the a99r2 is "Room 2".
Understand "Area 99 in Room 2" as a99r2.
the parent of the a99r2 is r2.
the X of the a99r2 is "547744.0".
the Y of the a99r2 is "1364800.0".

[create a7r3]
a7r3 is a area. "An area (7) in r3".
the printed name of the a7r3 is "Room 3".
Understand "Area 7 in Room 3" as a7r3.
the parent of the a7r3 is r3.
the X of the a7r3 is "548250.5167685038".
the Y of the a7r3 is "1366773.5133957588".

[create a8r3]
a8r3 is a area. "An area (8) in r3".
the printed name of the a8r3 is "Room 3".
Understand "Area 8 in Room 3" as a8r3.
the parent of the a8r3 is r3.
the X of the a8r3 is "548255.8708448358".
the Y of the a8r3 is "1366605.3162913683".

[create a13r3]
a13r3 is a area. "An area (13) in r3".
the printed name of the a13r3 is "Room 3".
Understand "Area 13 in Room 3" as a13r3.
the parent of the a13r3 is r3.
the X of the a13r3 is "548183.3333333333".
the Y of the a13r3 is "1366860.9694444444".

[create a50r3]
a50r3 is a area. "An area (50) in r3".
the printed name of the a50r3 is "Room 3".
Understand "Area 50 in Room 3" as a50r3.
the parent of the a50r3 is r3.
the X of the a50r3 is "548350.0".
the Y of the a50r3 is "1366904.0".

[create a51r3]
a51r3 is a area. "An area (51) in r3".
the printed name of the a51r3 is "Room 3".
Understand "Area 51 in Room 3" as a51r3.
the parent of the a51r3 is r3.
the X of the a51r3 is "548316.6666666667".
the Y of the a51r3 is "1366853.5364495006".

[create a101r3]
a101r3 is a area. "An area (101) in r3".
the printed name of the a101r3 is "Room 3".
Understand "Area 101 in Room 3" as a101r3.
the parent of the a101r3 is r3.
the X of the a101r3 is "548150.0".
the Y of the a101r3 is "1366904.0".

[create a17r4]
a17r4 is a area. "An area (17) in r4".
the printed name of the a17r4 is "Room 4".
Understand "Area 17 in Room 4" as a17r4.
the parent of the a17r4 is r4.
the X of the a17r4 is "548850.2390740267".
the Y of the a17r4 is "1364962.431282769".

[create a28r4]
a28r4 is a area. "An area (28) in r4".
the printed name of the a28r4 is "Room 4".
Understand "Area 28 in Room 4" as a28r4.
the parent of the a28r4 is r4.
the X of the a28r4 is "548872.2896516463".
the Y of the a28r4 is "1364960.8833195835".

[create a29r4]
a29r4 is a area. "An area (29) in r4".
the printed name of the a29r4 is "Room 4".
Understand "Area 29 in Room 4" as a29r4.
the parent of the a29r4 is r4.
the X of the a29r4 is "548966.4539705467".
the Y of the a29r4 is "1364848.7643334027".

[create a37r4]
a37r4 is a area. "An area (37) in r4".
the printed name of the a37r4 is "Room 4".
Understand "Area 37 in Room 4" as a37r4.
the parent of the a37r4 is r4.
the X of the a37r4 is "548754.7043713275".
the Y of the a37r4 is "1364822.7067112252".

[create a98r4]
a98r4 is a area. "An area (98) in r4".
the printed name of the a98r4 is "Room 4".
Understand "Area 98 in Room 4" as a98r4.
the parent of the a98r4 is r4.
the X of the a98r4 is "548724.0".
the Y of the a98r4 is "1364772.0".

[create a100r4]
a100r4 is a area. "An area (100) in r4".
the printed name of the a100r4 is "Room 4".
Understand "Area 100 in Room 4" as a100r4.
the parent of the a100r4 is r4.
the X of the a100r4 is "549007.0".
the Y of the a100r4 is "1364800.0".

[create a2r5]
a2r5 is a area. "An area (2) in r5".
the printed name of the a2r5 is "Room 5".
Understand "Area 2 in Room 5" as a2r5.
the parent of the a2r5 is r5.
the X of the a2r5 is "549382.9486065779".
the Y of the a2r5 is "1366778.1064398694".

[create a3r5]
a3r5 is a area. "An area (3) in r5".
the printed name of the a3r5 is "Room 5".
Understand "Area 3 in Room 5" as a3r5.
the parent of the a3r5 is r5.
the X of the a3r5 is "549712.5509290384".
the Y of the a3r5 is "1366861.0062088708".

[create a16r5]
a16r5 is a area. "An area (16) in r5".
the printed name of the a16r5 is "Room 5".
Understand "Area 16 in Room 5" as a16r5.
the parent of the a16r5 is r5.
the X of the a16r5 is "549048.4921500897".
the Y of the a16r5 is "1366860.7986347186".

[create a48r5]
a48r5 is a area. "An area (48) in r5".
the printed name of the a48r5 is "Room 5".
Understand "Area 48 in Room 5" as a48r5.
the parent of the a48r5 is r5.
the X of the a48r5 is "549769.0".
the Y of the a48r5 is "1366824.0".

[create a49r5]
a49r5 is a area. "An area (49) in r5".
the printed name of the a49r5 is "Room 5".
Understand "Area 49 in Room 5" as a49r5.
the parent of the a49r5 is r5.
the X of the a49r5 is "549728.662577747".
the Y of the a49r5 is "1366842.6912577746".

[create a58r5]
a58r5 is a area. "An area (58) in r5".
the printed name of the a58r5 is "Room 5".
Understand "Area 58 in Room 5" as a58r5.
the parent of the a58r5 is r5.
the X of the a58r5 is "549001.0".
the Y of the a58r5 is "1366914.0".

[create a59r5]
a59r5 is a area. "An area (59) in r5".
the printed name of the a59r5 is "Room 5".
Understand "Area 59 in Room 5" as a59r5.
the parent of the a59r5 is r5.
the X of the a59r5 is "549031.9153780069".
the Y of the a59r5 is "1366888.1834621993".

[create a62r5]
a62r5 is a area. "An area (62) in r5".
the printed name of the a62r5 is "Room 5".
Understand "Area 62 in Room 5" as a62r5.
the parent of the a62r5 is r5.
the X of the a62r5 is "549760.0".
the Y of the a62r5 is "1366914.0".

[create a63r5]
a63r5 is a area. "An area (63) in r5".
the printed name of the a63r5 is "Room 5".
Understand "Area 63 in Room 5" as a63r5.
the parent of the a63r5 is r5.
the X of the a63r5 is "549728.9123931623".
the Y of the a63r5 is "1366888.1662393163".

[create a78r5]
a78r5 is a area. "An area (78) in r5".
the printed name of the a78r5 is "Room 5".
Understand "Area 78 in Room 5" as a78r5.
the parent of the a78r5 is r5.
the X of the a78r5 is "548992.0".
the Y of the a78r5 is "1366824.0".

[create a79r5]
a79r5 is a area. "An area (79) in r5".
the printed name of the a79r5 is "Room 5".
Understand "Area 79 in Room 5" as a79r5.
the parent of the a79r5 is r5.
the X of the a79r5 is "549032.7622699386".
the Y of the a79r5 is "1366842.6487730062".

[create a19r6]
a19r6 is a area. "An area (19) in r6".
the printed name of the a19r6 is "Room 6".
Understand "Area 19 in Room 6" as a19r6.
the parent of the a19r6 is r6.
the X of the a19r6 is "546634.69417693".
the Y of the a19r6 is "1364998.5748196018".

[create a24r6]
a24r6 is a area. "An area (24) in r6".
the printed name of the a24r6 is "Room 6".
Understand "Area 24 in Room 6" as a24r6.
the parent of the a24r6 is r6.
the X of the a24r6 is "546968.0529104449".
the Y of the a24r6 is "1364957.1418691212".

[create a25r6]
a25r6 is a area. "An area (25) in r6".
the printed name of the a25r6 is "Room 6".
Understand "Area 25 in Room 6" as a25r6.
the parent of the a25r6 is r6.
the X of the a25r6 is "547075.7337947071".
the Y of the a25r6 is "1364883.3125148227".

[create a30r6]
a30r6 is a area. "An area (30) in r6".
the printed name of the a30r6 is "Room 6".
Understand "Area 30 in Room 6" as a30r6.
the parent of the a30r6 is r6.
the X of the a30r6 is "546683.1759575147".
the Y of the a30r6 is "1364931.45737941".

[create a33r6]
a33r6 is a area. "An area (33) in r6".
the printed name of the a33r6 is "Room 6".
Understand "Area 33 in Room 6" as a33r6.
the parent of the a33r6 is r6.
the X of the a33r6 is "547337.7817693062".
the Y of the a33r6 is "1364844.9748289345".

[create a64r6]
a64r6 is a area. "An area (64) in r6".
the printed name of the a64r6 is "Room 6".
Understand "Area 64 in Room 6" as a64r6.
the parent of the a64r6 is r6.
the X of the a64r6 is "546613.0".
the Y of the a64r6 is "1364772.0".

[create a65r6]
a65r6 is a area. "An area (65) in r6".
the printed name of the a65r6 is "Room 6".
Understand "Area 65 in Room 6" as a65r6.
the parent of the a65r6 is r6.
the X of the a65r6 is "546648.1492196214".
the Y of the a65r6 is "1364813.685355934".

[create a66r6]
a66r6 is a area. "An area (66) in r6".
the printed name of the a66r6 is "Room 6".
Understand "Area 66 in Room 6" as a66r6.
the parent of the a66r6 is r6.
the X of the a66r6 is "547395.0".
the Y of the a66r6 is "1364880.0".

[create a68r6]
a68r6 is a area. "An area (68) in r6".
the printed name of the a68r6 is "Room 6".
Understand "Area 68 in Room 6" as a68r6.
the parent of the a68r6 is r6.
the X of the a68r6 is "546535.0".
the Y of the a68r6 is "1364880.0".

[create a69r6]
a69r6 is a area. "An area (69) in r6".
the printed name of the a69r6 is "Room 6".
Understand "Area 69 in Room 6" as a69r6.
the parent of the a69r6 is r6.
the X of the a69r6 is "546526.9727817976".
the Y of the a69r6 is "1364920.869285258".

[create a84r6]
a84r6 is a area. "An area (84) in r6".
the printed name of the a84r6 is "Room 6".
Understand "Area 84 in Room 6" as a84r6.
the parent of the a84r6 is r6.
the X of the a84r6 is "547386.0".
the Y of the a84r6 is "1364791.0".

[create a85r6]
a85r6 is a area. "An area (85) in r6".
the printed name of the a85r6 is "Room 6".
Understand "Area 85 in Room 6" as a85r6.
the parent of the a85r6 is r6.
the X of the a85r6 is "547354.7862838916".
the Y of the a85r6 is "1364816.633971292".

[create a88r6]
a88r6 is a area. "An area (88) in r6".
the printed name of the a88r6 is "Room 6".
Understand "Area 88 in Room 6" as a88r6.
the parent of the a88r6 is r6.
the X of the a88r6 is "547352.1485252038".
the Y of the a88r6 is "1364861.8557671143".

[create a89r6]
a89r6 is a area. "An area (89) in r6".
the printed name of the a89r6 is "Room 6".
Understand "Area 89 in Room 6" as a89r6.
the parent of the a89r6 is r6.
the X of the a89r6 is "546535.0818360213".
the Y of the a89r6 is "1364880.0161587049".

[create a90r6]
a90r6 is a area. "An area (90) in r6".
the printed name of the a90r6 is "Room 6".
Understand "Area 90 in Room 6" as a90r6.
the parent of the a90r6 is r6.
the X of the a90r6 is "546539.3930352519".
the Y of the a90r6 is "1364857.96671143".

[create a91r6]
a91r6 is a area. "An area (91) in r6".
the printed name of the a91r6 is "Room 6".
Understand "Area 91 in Room 6" as a91r6.
the parent of the a91r6 is r6.
the X of the a91r6 is "546544.0".
the Y of the a91r6 is "1364791.0".

[create a92r6]
a92r6 is a area. "An area (92) in r6".
the printed name of the a92r6 is "Room 6".
Understand "Area 92 in Room 6" as a92r6.
the parent of the a92r6 is r6.
the X of the a92r6 is "546569.646350374".
the Y of the a92r6 is "1364816.0709792513".

[create a93r6]
a93r6 is a area. "An area (93) in r6".
the printed name of the a93r6 is "Room 6".
Understand "Area 93 in Room 6" as a93r6.
the parent of the a93r6 is r6.
the X of the a93r6 is "547126.9101123596".
the Y of the a93r6 is "1364800.0".

[create a94r6]
a94r6 is a area. "An area (94) in r6".
the printed name of the a94r6 is "Room 6".
Understand "Area 94 in Room 6" as a94r6.
the parent of the a94r6 is r6.
the X of the a94r6 is "547088.9999302698".
the Y of the a94r6 is "1364799.2875970025".

[create a102r6]
a102r6 is a area. "An area (102) in r6".
the printed name of the a102r6 is "Room 6".
Understand "Area 102 in Room 6" as a102r6.
the parent of the a102r6 is r6.
the X of the a102r6 is "546604.6901746606".
the Y of the a102r6 is "1364864.4473575065".

[create a103r6]
a103r6 is a area. "An area (103) in r6".
the printed name of the a103r6 is "Room 6".
Understand "Area 103 in Room 6" as a103r6.
the parent of the a103r6 is r6.
the X of the a103r6 is "546668.6300804161".
the Y of the a103r6 is "1364910.570827561".

[create d0]
d0 is a door. "A door between a28r4 and a2r5".
d0 is north of a28r4 and south of a2r5.

[create d2]
d2 is a door. "A door between a18r1 and a19r6".
d2 is southeast of a18r1 and northwest of a19r6.

[create d3]
d3 is a door. "A door between a22r1 and a24r6".
d3 is south of a22r1 and north of a24r6.

[create d4]
d4 is a door. "A door between a26r2 and a8r3".
d4 is north of a26r2 and south of a8r3.

[create d6]
d6 is a door. "A door between a23r1 and a31r2".
d6 is south of a23r1 and north of a31r2.

[create d8]
d8 is a door. "A door between a8r3 and a17r4".
d8 is southeast of a8r3 and northwest of a17r4.

[create d12]
d12 is a door. "A door between a20r0 and a18r1".
d12 is northeast of a20r0 and southwest of a18r1.

[create d13]
d13 is a door. "A door between a20r0 and a10r1".
d13 is north of a20r0 and south of a10r1.

southeast of a0r1 is northwest of a1r1.

east of a2r5 is west of a3r5.

west of a0r1 is east of a4r1.

east of a5r1 is west of a6r1.

south of a7r3 is north of a8r3.

east of a9r1 is west of a10r1.

east of a10r1 is west of a11r1.

southeast of a1r1 is northwest of a12r1.

southwest of a5r1 is northeast of a12r1.

northwest of a7r3 is southeast of a13r3.

east of a14r1 is west of a15r1.

west of a2r5 is east of a16r5.

south of a20r0 is north of a21r0.

southwest of a4r1 is northeast of a18r1.

southeast of a22r1 is northwest of a23r1.

south of a12r1 is north of a22r1.

southeast of a24r6 is northwest of a25r6.

southeast of a26r2 is northwest of a27r2.

southeast of a28r4 is northwest of a29r4.

southeast of a19r6 is northwest of a30r6.

east of a21r0 is west of a32r0.

east of a25r6 is west of a33r6.

west of a21r0 is east of a34r0.

northeast of a35r2 is southwest of a36r2.

west of a24r6 is east of a30r6.

southwest of a17r4 is northeast of a37r4.

southwest of a38r1 is northeast of a39r1.

west of a40r1 is east of a41r1.

east of a42r1 is west of a43r1.

west of a44r1 is east of a45r1.

east of a46r1 is west of a47r1.

northwest of a48r5 is southeast of a49r5.

southwest of a50r3 is northeast of a51r3.

northwest of a52r1 is southeast of a53r1.

southeast of a54r1 is northwest of a55r1.

southeast of a56r1 is northwest of a57r1.

southeast of a58r5 is northwest of a59r5.

southwest of a60r1 is northeast of a61r1.

southwest of a62r5 is northeast of a63r5.

northeast of a64r6 is southwest of a65r6.

north of a68r6 is south of a69r6.

east of a70r1 is west of a71r1.

northeast of a72r1 is southwest of a73r1.

northeast of a74r1 is southwest of a75r1.

northwest of a76r1 is southeast of a77r1.

northeast of a78r5 is southwest of a79r5.

southeast of a80r0 is northwest of a81r0.

northwest of a82r0 is southeast of a83r0.

northwest of a84r6 is southeast of a85r6.

southwest of a86r0 is northeast of a87r0.

southwest of a66r6 is northeast of a88r6.

south of a89r6 is north of a90r6.

northeast of a91r6 is southwest of a92r6.

west of a93r6 is east of a94r6.

northeast of a95r0 is southwest of a96r0.

southwest of a35r2 is northeast of a97r2.

southwest of a37r4 is northeast of a98r4.

southeast of a27r2 is northwest of a99r2.

southeast of a29r4 is northwest of a100r4.

northwest of a13r3 is southeast of a101r3.

southwest of a34r0 is northeast of a96r0.

northwest of a34r0 is southeast of a81r0.

southeast of a32r0 is northwest of a83r0.

southeast of a33r6 is northwest of a85r6.

northeast of a32r0 is southwest of a87r0.

northeast of a33r6 is southwest of a88r6.

east of a90r6 is west of a102r6.

northeast of a3r5 is southwest of a63r5.

northeast of a6r1 is southwest of a39r1.

north of a1r1 is south of a41r1.

north of a0r1 is south of a43r1.

north of a5r1 is south of a45r1.

north of a15r1 is south of a47r1.

southwest of a16r5 is northeast of a79r5.

southeast of a3r5 is northwest of a49r5.

northeast of a7r3 is southwest of a51r3.

southeast of a6r1 is northwest of a53r1.

northwest of a9r1 is southeast of a55r1.

northwest of a14r1 is southeast of a57r1.

northwest of a16r5 is southeast of a59r5.

northeast of a11r1 is southwest of a61r1.

southwest of a31r2 is northeast of a36r2.

west of a26r2 is east of a31r2.

southwest of a19r6 is northeast of a69r6.

west of a23r1 is east of a71r1.

northwest of a4r1 is southeast of a15r1.

southwest of a14r1 is northeast of a73r1.

southwest of a9r1 is northeast of a75r1.

southeast of a11r1 is northwest of a77r1.

northeast of a92r6 is southwest of a102r6.

south of a25r6 is north of a94r6.

northeast of a102r6 is southwest of a103r6.

north of a65r6 is south of a103r6.

west of a36r2 is east of a67r2.

southwest of a30r6 is northeast of a103r6.

east of a17r4 is west of a28r4.

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
the player is in a72r1.

the orientation of the player is 6.
