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

[create a29r0]
a29r0 is a area. "An area (29) in r0".
the printed name of the a29r0 is "Room 0".
Understand "Area 29 in Room 0" as a29r0.
the parent of the a29r0 is r0.
the X of the a29r0 is "548266.9912057456".
the Y of the a29r0 is "1664564.2196130909".

[create a30r0]
a30r0 is a area. "An area (30) in r0".
the printed name of the a30r0 is "Room 0".
Understand "Area 30 in Room 0" as a30r0.
the parent of the a30r0 is r0.
the X of the a30r0 is "548300.5483636888".
the Y of the a30r0 is "1664566.9119265778".

[create a31r0]
a31r0 is a area. "An area (31) in r0".
the printed name of the a31r0 is "Room 0".
Understand "Area 31 in Room 0" as a31r0.
the parent of the a31r0 is r0.
the X of the a31r0 is "548105.686594203".
the Y of the a31r0 is "1664689.6751110335".

[create a50r0]
a50r0 is a area. "An area (50) in r0".
the printed name of the a50r0 is "Room 0".
Understand "Area 50 in Room 0" as a50r0.
the parent of the a50r0 is r0.
the X of the a50r0 is "548248.0".
the Y of the a50r0 is "1664453.0".

[create a51r0]
a51r0 is a area. "An area (51) in r0".
the printed name of the a51r0 is "Room 0".
Understand "Area 51 in Room 0" as a51r0.
the parent of the a51r0 is r0.
the X of the a51r0 is "548234.0".
the Y of the a51r0 is "1664514.4050772628".

[create a52r0]
a52r0 is a area. "An area (52) in r0".
the printed name of the a52r0 is "Room 0".
Understand "Area 52 in Room 0" as a52r0.
the parent of the a52r0 is r0.
the X of the a52r0 is "548334.0".
the Y of the a52r0 is "1664453.0".

[create a53r0]
a53r0 is a area. "An area (53) in r0".
the printed name of the a53r0 is "Room 0".
Understand "Area 53 in Room 0" as a53r0.
the parent of the a53r0 is r0.
the X of the a53r0 is "548312.5".
the Y of the a53r0 is "1664538.00390625".

[create a80r0]
a80r0 is a area. "An area (80) in r0".
the printed name of the a80r0 is "Room 0".
Understand "Area 80 in Room 0" as a80r0.
the parent of the a80r0 is r0.
the X of the a80r0 is "548107.0".
the Y of the a80r0 is "1664739.0".

[create a0r1]
a0r1 is a area. "An area (0) in r1".
the printed name of the a0r1 is "Room 1".
Understand "Area 0 in Room 1" as a0r1.
the parent of the a0r1 is r1.
the X of the a0r1 is "548382.8810340974".
the Y of the a0r1 is "1667028.0634808128".

[create a1r1]
a1r1 is a area. "An area (1) in r1".
the printed name of the a1r1 is "Room 1".
Understand "Area 1 in Room 1" as a1r1.
the parent of the a1r1 is r1.
the X of the a1r1 is "548452.0662053295".
the Y of the a1r1 is "1666979.1879088343".

[create a5r1]
a5r1 is a area. "An area (5) in r1".
the printed name of the a5r1 is "Room 1".
Understand "Area 5 in Room 1" as a5r1.
the parent of the a5r1 is r1.
the X of the a5r1 is "548097.5080993522".
the Y of the a5r1 is "1667064.2282217424".

[create a12r1]
a12r1 is a area. "An area (12) in r1".
the printed name of the a12r1 is "Room 1".
Understand "Area 12 in Room 1" as a12r1.
the parent of the a12r1 is r1.
the X of the a12r1 is "548771.6489868986".
the Y of the a12r1 is "1666972.4532954688".

[create a13r1]
a13r1 is a area. "An area (13) in r1".
the printed name of the a13r1 is "Room 1".
Understand "Area 13 in Room 1" as a13r1.
the parent of the a13r1 is r1.
the X of the a13r1 is "549106.1865386874".
the Y of the a13r1 is "1667059.6370670425".

[create a14r1]
a14r1 is a area. "An area (14) in r1".
the printed name of the a14r1 is "Room 1".
Understand "Area 14 in Room 1" as a14r1.
the parent of the a14r1 is r1.
the X of the a14r1 is "548745.897977266".
the Y of the a14r1 is "1666695.274434491".

[create a15r1]
a15r1 is a area. "An area (15) in r1".
the printed name of the a15r1 is "Room 1".
Understand "Area 15 in Room 1" as a15r1.
the parent of the a15r1 is r1.
the X of the a15r1 is "548744.5585798462".
the Y of the a15r1 is "1666767.7486609586".

[create a16r1]
a16r1 is a area. "An area (16) in r1".
the printed name of the a16r1 is "Room 1".
Understand "Area 16 in Room 1" as a16r1.
the parent of the a16r1 is r1.
the X of the a16r1 is "548454.7657499192".
the Y of the a16r1 is "1666864.6345861065".

[create a17r1]
a17r1 is a area. "An area (17) in r1".
the printed name of the a17r1 is "Room 1".
Understand "Area 17 in Room 1" as a17r1.
the parent of the a17r1 is r1.
the X of the a17r1 is "548456.4206104914".
the Y of the a17r1 is "1666798.199910067".

[create a18r1]
a18r1 is a area. "An area (18) in r1".
the printed name of the a18r1 is "Room 1".
Understand "Area 18 in Room 1" as a18r1.
the parent of the a18r1 is r1.
the X of the a18r1 is "548744.2822228307".
the Y of the a18r1 is "1666827.9152381327".

[create a19r1]
a19r1 is a area. "An area (19) in r1".
the printed name of the a19r1 is "Room 1".
Understand "Area 19 in Room 1" as a19r1.
the parent of the a19r1 is r1.
the X of the a19r1 is "548744.5".
the Y of the a19r1 is "1666946.361338798".

[create a28r1]
a28r1 is a area. "An area (28) in r1".
the printed name of the a28r1 is "Room 1".
Understand "Area 28 in Room 1" as a28r1.
the parent of the a28r1 is r1.
the X of the a28r1 is "548458.0808113122".
the Y of the a28r1 is "1666731.764975888".

[create a40r1]
a40r1 is a area. "An area (40) in r1".
the printed name of the a40r1 is "Room 1".
Understand "Area 40 in Room 1" as a40r1.
the parent of the a40r1 is r1.
the X of the a40r1 is "549164.0".
the Y of the a40r1 is "1667024.0".

[create a41r1]
a41r1 is a area. "An area (41) in r1".
the printed name of the a41r1 is "Room 1".
Understand "Area 41 in Room 1" as a41r1.
the parent of the a41r1 is r1.
the X of the a41r1 is "549121.1798056157".
the Y of the a41r1 is "1667042.019978402".

[create a58r1]
a58r1 is a area. "An area (58) in r1".
the printed name of the a58r1 is "Room 1".
Understand "Area 58 in Room 1" as a58r1.
the parent of the a58r1 is r1.
the X of the a58r1 is "548818.1111111111".
the Y of the a58r1 is "1667104.0".

[create a59r1]
a59r1 is a area. "An area (59) in r1".
the printed name of the a59r1 is "Room 1".
Understand "Area 59 in Room 1" as a59r1.
the parent of the a59r1 is r1.
the X of the a59r1 is "548785.1923615087".
the Y of the a59r1 is "1667105.4040895503".

[create a64r1]
a64r1 is a area. "An area (64) in r1".
the printed name of the a64r1 is "Room 1".
Understand "Area 64 in Room 1" as a64r1.
the parent of the a64r1 is r1.
the X of the a64r1 is "548342.0".
the Y of the a64r1 is "1667104.0".

[create a65r1]
a65r1 is a area. "An area (65) in r1".
the printed name of the a65r1 is "Room 1".
Understand "Area 65 in Room 1" as a65r1.
the parent of the a65r1 is r1.
the X of the a65r1 is "548374.9940170939".
the Y of the a65r1 is "1667105.7505982905".

[create a66r1]
a66r1 is a area. "An area (66) in r1".
the printed name of the a66r1 is "Room 1".
Understand "Area 66 in Room 1" as a66r1.
the parent of the a66r1 is r1.
the X of the a66r1 is "548043.0".
the Y of the a66r1 is "1667024.0".

[create a67r1]
a67r1 is a area. "An area (67) in r1".
the printed name of the a67r1 is "Room 1".
Understand "Area 67 in Room 1" as a67r1.
the parent of the a67r1 is r1.
the X of the a67r1 is "548070.2059471365".
the Y of the a67r1 is "1667044.0044052864".

[create a68r1]
a68r1 is a area. "An area (68) in r1".
the printed name of the a68r1 is "Room 1".
Understand "Area 68 in Room 1" as a68r1.
the parent of the a68r1 is r1.
the X of the a68r1 is "549154.0".
the Y of the a68r1 is "1667114.0".

[create a69r1]
a69r1 is a area. "An area (69) in r1".
the printed name of the a69r1 is "Room 1".
Understand "Area 69 in Room 1" as a69r1.
the parent of the a69r1 is r1.
the X of the a69r1 is "549123.1050372064".
the Y of the a69r1 is "1667087.7894485786".

[create a76r1]
a76r1 is a area. "An area (76) in r1".
the printed name of the a76r1 is "Room 1".
Understand "Area 76 in Room 1" as a76r1.
the parent of the a76r1 is r1.
the X of the a76r1 is "548052.0".
the Y of the a76r1 is "1667114.0".

[create a77r1]
a77r1 is a area. "An area (77) in r1".
the printed name of the a77r1 is "Room 1".
Understand "Area 77 in Room 1" as a77r1.
the parent of the a77r1 is r1.
the X of the a77r1 is "548083.0876068376".
the Y of the a77r1 is "1667088.1662393163".

[create a22r2]
a22r2 is a area. "An area (22) in r2".
the printed name of the a22r2 is "Room 2".
Understand "Area 22 in Room 2" as a22r2.
the parent of the a22r2 is r2.
the X of the a22r2 is "548079.0218346365".
the Y of the a22r2 is "1665084.5039537789".

[create a23r2]
a23r2 is a area. "An area (23) in r2".
the printed name of the a23r2 is "Room 2".
Understand "Area 23 in Room 2" as a23r2.
the parent of the a23r2 is r2.
the X of the a23r2 is "548410.3853736456".
the Y of the a23r2 is "1665447.6808904756".

[create a27r2]
a27r2 is a area. "An area (27) in r2".
the printed name of the a27r2 is "Room 2".
Understand "Area 27 in Room 2" as a27r2.
the parent of the a27r2 is r2.
the X of the a27r2 is "547051.822417243".
the Y of the a27r2 is "1665357.5305011156".

[create a32r2]
a32r2 is a area. "An area (32) in r2".
the printed name of the a32r2 is "Room 2".
Understand "Area 32 in Room 2" as a32r2.
the parent of the a32r2 is r2.
the X of the a32r2 is "546885.0".
the Y of the a32r2 is "1665273.0".

[create a33r2]
a33r2 is a area. "An area (33) in r2".
the printed name of the a33r2 is "Room 2".
Understand "Area 33 in Room 2" as a33r2.
the parent of the a33r2 is r2.
the X of the a33r2 is "546955.9942396785".
the Y of the a33r2 is "1665311.6157052466".

[create a34r2]
a34r2 is a area. "An area (34) in r2".
the printed name of the a34r2 is "Room 2".
Understand "Area 34 in Room 2" as a34r2.
the parent of the a34r2 is r2.
the X of the a34r2 is "548532.0".
the Y of the a34r2 is "1665399.0".

[create a35r2]
a35r2 is a area. "An area (35) in r2".
the printed name of the a35r2 is "Room 2".
Understand "Area 35 in Room 2" as a35r2.
the parent of the a35r2 is r2.
the X of the a35r2 is "548435.4006425084".
the Y of the a35r2 is "1665437.4374839372".

[create a54r2]
a54r2 is a area. "An area (54) in r2".
the printed name of the a54r2 is "Room 2".
Understand "Area 54 in Room 2" as a54r2.
the parent of the a54r2 is r2.
the X of the a54r2 is "548132.0".
the Y of the a54r2 is "1664963.0".

[create a55r2]
a55r2 is a area. "An area (55) in r2".
the printed name of the a55r2 is "Room 2".
Understand "Area 55 in Room 2" as a55r2.
the parent of the a55r2 is r2.
the X of the a55r2 is "548096.4166666667".
the Y of the a55r2 is "1665045.8310397551".

[create a2r3]
a2r3 is a area. "An area (2) in r3".
the printed name of the a2r3 is "Room 3".
Understand "Area 2 in Room 3" as a2r3.
the parent of the a2r3 is r3.
the X of the a2r3 is "546654.8405919939".
the Y of the a2r3 is "1667027.228865128".

[create a3r3]
a3r3 is a area. "An area (3) in r3".
the printed name of the a3r3 is "Room 3".
Understand "Area 3 in Room 3" as a3r3.
the parent of the a3r3 is r3.
the X of the a3r3 is "546754.4949478189".
the Y of the a3r3 is "1666970.4230603536".

[create a4r3]
a4r3 is a area. "An area (4) in r3".
the printed name of the a4r3 is "Room 3".
Understand "Area 4 in Room 3" as a4r3.
the parent of the a4r3 is r3.
the X of the a4r3 is "547018.1344309739".
the Y of the a4r3 is "1666937.9437927655".

[create a6r3]
a6r3 is a area. "An area (6) in r3".
the printed name of the a6r3 is "Room 3".
Understand "Area 6 in Room 3" as a6r3.
the parent of the a6r3 is r3.
the X of the a6r3 is "547079.6590106003".
the Y of the a6r3 is "1666993.9829210837".

[create a7r3]
a7r3 is a area. "An area (7) in r3".
the printed name of the a7r3 is "Room 3".
Understand "Area 7 in Room 3" as a7r3.
the parent of the a7r3 is r3.
the X of the a7r3 is "547377.8606168446".
the Y of the a7r3 is "1667059.860290629".

[create a11r3]
a11r3 is a area. "An area (11) in r3".
the printed name of the a11r3 is "Room 3".
Understand "Area 11 in Room 3" as a11r3.
the parent of the a11r3 is r3.
the X of the a11r3 is "546411.5405048297".
the Y of the a11r3 is "1667060.8544286496".

[create a20r3]
a20r3 is a area. "An area (20) in r3".
the printed name of the a20r3 is "Room 3".
Understand "Area 20 in Room 3" as a20r3.
the parent of the a20r3 is r3.
the X of the a20r3 is "547014.2117752554".
the Y of the a20r3 is "1666844.978948947".

[create a21r3]
a21r3 is a area. "An area (21) in r3".
the printed name of the a21r3 is "Room 3".
Understand "Area 21 in Room 3" as a21r3.
the parent of the a21r3 is r3.
the X of the a21r3 is "546739.9934804996".
the Y of the a21r3 is "1666875.0001716458".

[create a36r3]
a36r3 is a area. "An area (36) in r3".
the printed name of the a36r3 is "Room 3".
Understand "Area 36 in Room 3" as a36r3.
the parent of the a36r3 is r3.
the X of the a36r3 is "546855.0".
the Y of the a36r3 is "1666874.0".

[create a37r3]
a37r3 is a area. "An area (37) in r3".
the printed name of the a37r3 is "Room 3".
Understand "Area 37 in Room 3" as a37r3.
the parent of the a37r3 is r3.
the X of the a37r3 is "546840.8974996369".
the Y of the a37r3 is "1666837.3831533114".

[create a38r3]
a38r3 is a area. "An area (38) in r3".
the printed name of the a38r3 is "Room 3".
Understand "Area 38 in Room 3" as a38r3.
the parent of the a38r3 is r3.
the X of the a38r3 is "546857.3187022901".
the Y of the a38r3 is "1666879.379389313".

[create a39r3]
a39r3 is a area. "An area (39) in r3".
the printed name of the a39r3 is "Room 3".
Understand "Area 39 in Room 3" as a39r3.
the parent of the a39r3 is r3.
the X of the a39r3 is "546934.4038944134".
the Y of the a39r3 is "1666842.9635984208".

[create a42r3]
a42r3 is a area. "An area (42) in r3".
the printed name of the a42r3 is "Room 3".
Understand "Area 42 in Room 3" as a42r3.
the parent of the a42r3 is r3.
the X of the a42r3 is "547435.0".
the Y of the a42r3 is "1667024.0".

[create a43r3]
a43r3 is a area. "An area (43) in r3".
the printed name of the a43r3 is "Room 3".
Understand "Area 43 in Room 3" as a43r3.
the parent of the a43r3 is r3.
the X of the a43r3 is "547392.94".
the Y of the a43r3 is "1667042.519".

[create a46r3]
a46r3 is a area. "An area (46) in r3".
the printed name of the a46r3 is "Room 3".
Understand "Area 46 in Room 3" as a46r3.
the parent of the a46r3 is r3.
the X of the a46r3 is "546355.0".
the Y of the a46r3 is "1667024.0".

[create a47r3]
a47r3 is a area. "An area (47) in r3".
the printed name of the a47r3 is "Room 3".
Understand "Area 47 in Room 3" as a47r3.
the parent of the a47r3 is r3.
the X of the a47r3 is "546395.7622699386".
the Y of the a47r3 is "1667042.6487730062".

[create a60r3]
a60r3 is a area. "An area (60) in r3".
the printed name of the a60r3 is "Room 3".
Understand "Area 60 in Room 3" as a60r3.
the parent of the a60r3 is r3.
the X of the a60r3 is "546615.0".
the Y of the a60r3 is "1667114.0".

[create a61r3]
a61r3 is a area. "An area (61) in r3".
the printed name of the a61r3 is "Room 3".
Understand "Area 61 in Room 3" as a61r3.
the parent of the a61r3 is r3.
the X of the a61r3 is "546646.8264604809".
the Y of the a61r3 is "1667105.767353952".

[create a62r3]
a62r3 is a area. "An area (62) in r3".
the printed name of the a62r3 is "Room 3".
Understand "Area 62 in Room 3" as a62r3.
the parent of the a62r3 is r3.
the X of the a62r3 is "547124.0".
the Y of the a62r3 is "1667104.0".

[create a63r3]
a63r3 is a area. "An area (63) in r3".
the printed name of the a63r3 is "Room 3".
Understand "Area 63 in Room 3" as a63r3.
the parent of the a63r3 is r3.
the X of the a63r3 is "547091.005982906".
the Y of the a63r3 is "1667105.7505982905".

[create a74r3]
a74r3 is a area. "An area (74) in r3".
the printed name of the a74r3 is "Room 3".
Understand "Area 74 in Room 3" as a74r3.
the parent of the a74r3 is r3.
the X of the a74r3 is "547426.0".
the Y of the a74r3 is "1667114.0".

[create a75r3]
a75r3 is a area. "An area (75) in r3".
the printed name of the a75r3 is "Room 3".
Understand "Area 75 in Room 3" as a75r3.
the parent of the a75r3 is r3.
the X of the a75r3 is "547394.9123931623".
the Y of the a75r3 is "1667088.1662393163".

[create a78r3]
a78r3 is a area. "An area (78) in r3".
the printed name of the a78r3 is "Room 3".
Understand "Area 78 in Room 3" as a78r3.
the parent of the a78r3 is r3.
the X of the a78r3 is "546364.0".
the Y of the a78r3 is "1667114.0".

[create a79r3]
a79r3 is a area. "An area (79) in r3".
the printed name of the a79r3 is "Room 3".
Understand "Area 79 in Room 3" as a79r3.
the parent of the a79r3 is r3.
the X of the a79r3 is "546395.0876068379".
the Y of the a79r3 is "1667088.1662393163".

[create a24r4]
a24r4 is a area. "An area (24) in r4".
the printed name of the a24r4 is "Room 4".
Understand "Area 24 in Room 4" as a24r4.
the parent of the a24r4 is r4.
the X of the a24r4 is "546015.6294029708".
the Y of the a24r4 is "1665133.519746642".

[create a25r4]
a25r4 is a area. "An area (25) in r4".
the printed name of the a25r4 is "Room 4".
Understand "Area 25 in Room 4" as a25r4.
the parent of the a25r4 is r4.
the X of the a25r4 is "546106.9629832632".
the Y of the a25r4 is "1665128.2779362001".

[create a26r4]
a26r4 is a area. "An area (26) in r4".
the printed name of the a26r4 is "Room 4".
Understand "Area 26 in Room 4" as a26r4.
the parent of the a26r4 is r4.
the X of the a26r4 is "545945.0613114523".
the Y of the a26r4 is "1665024.5596518875".

[create a56r4]
a56r4 is a area. "An area (56) in r4".
the printed name of the a56r4 is "Room 4".
Understand "Area 56 in Room 4" as a56r4.
the parent of the a56r4 is r4.
the X of the a56r4 is "546213.0".
the Y of the a56r4 is "1665000.0".

[create a57r4]
a57r4 is a area. "An area (57) in r4".
the printed name of the a57r4 is "Room 4".
Understand "Area 57 in Room 4" as a57r4.
the parent of the a57r4 is r4.
the X of the a57r4 is "546170.4517163367".
the Y of the a57r4 is "1665049.4252993807".

[create a81r4]
a81r4 is a area. "An area (81) in r4".
the printed name of the a81r4 is "Room 4".
Understand "Area 81 in Room 4" as a81r4.
the parent of the a81r4 is r4.
the X of the a81r4 is "545912.0".
the Y of the a81r4 is "1664972.0".

[create a8r5]
a8r5 is a area. "An area (8) in r5".
the printed name of the a8r5 is "Room 5".
Understand "Area 8 in Room 5" as a8r5.
the parent of the a8r5 is r5.
the X of the a8r5 is "545050.281371821".
the Y of the a8r5 is "1667061.2845894436".

[create a9r5]
a9r5 is a area. "An area (9) in r5".
the printed name of the a9r5 is "Room 5".
Understand "Area 9 in Room 5" as a9r5.
the parent of the a9r5 is r5.
the X of the a9r5 is "545400.6942110014".
the Y of the a9r5 is "1666948.1635622133".

[create a10r5]
a10r5 is a area. "An area (10) in r5".
the printed name of the a10r5 is "Room 5".
Understand "Area 10 in Room 5" as a10r5.
the parent of the a10r5 is r5.
the X of the a10r5 is "545769.6927655565".
the Y of the a10r5 is "1667061.104115366".

[create a44r5]
a44r5 is a area. "An area (44) in r5".
the printed name of the a44r5 is "Room 5".
Understand "Area 44 in Room 5" as a44r5.
the parent of the a44r5 is r5.
the X of the a44r5 is "545826.0".
the Y of the a44r5 is "1667024.0".

[create a45r5]
a45r5 is a area. "An area (45) in r5".
the printed name of the a45r5 is "Room 5".
Understand "Area 45 in Room 5" as a45r5.
the parent of the a45r5 is r5.
the X of the a45r5 is "545785.5982564336".
the Y of the a45r5 is "1667042.6848256432".

[create a48r5]
a48r5 is a area. "An area (48) in r5".
the printed name of the a48r5 is "Room 5".
Understand "Area 48 in Room 5" as a48r5.
the parent of the a48r5 is r5.
the X of the a48r5 is "544994.0".
the Y of the a48r5 is "1667024.0".

[create a49r5]
a49r5 is a area. "An area (49) in r5".
the printed name of the a49r5 is "Room 5".
Understand "Area 49 in Room 5" as a49r5.
the parent of the a49r5 is r5.
the X of the a49r5 is "545033.4941364606".
the Y of the a49r5 is "1667042.7755863538".

[create a70r5]
a70r5 is a area. "An area (70) in r5".
the printed name of the a70r5 is "Room 5".
Understand "Area 70 in Room 5" as a70r5.
the parent of the a70r5 is r5.
the X of the a70r5 is "545817.0".
the Y of the a70r5 is "1667114.0".

[create a71r5]
a71r5 is a area. "An area (71) in r5".
the printed name of the a71r5 is "Room 5".
Understand "Area 71 in Room 5" as a71r5.
the parent of the a71r5 is r5.
the X of the a71r5 is "545786.0846219931".
the Y of the a71r5 is "1667088.1834621995".

[create a72r5]
a72r5 is a area. "An area (72) in r5".
the printed name of the a72r5 is "Room 5".
Understand "Area 72 in Room 5" as a72r5.
the parent of the a72r5 is r5.
the X of the a72r5 is "545003.0".
the Y of the a72r5 is "1667114.0".

[create a73r5]
a73r5 is a area. "An area (73) in r5".
the printed name of the a73r5 is "Room 5".
Understand "Area 73 in Room 5" as a73r5.
the parent of the a73r5 is r5.
the X of the a73r5 is "545034.0876068377".
the Y of the a73r5 is "1667088.1662393163".

[create d0]
d0 is a door. "A door between a30r0 and a14r1".
d0 is north of a30r0 and south of a14r1.

[create d2]
d2 is a door. "A door between a21r3 and a25r4".
d2 is south of a21r3 and north of a25r4.

[create d4]
d4 is a door. "A door between a24r4 and a9r5".
d4 is north of a24r4 and south of a9r5.

[create d6]
d6 is a door. "A door between a27r2 and a20r3".
d6 is north of a27r2 and south of a20r3.

[create d8]
d8 is a door. "A door between a28r1 and a23r2".
d8 is south of a28r1 and north of a23r2.

southeast of a0r1 is northwest of a1r1.

southeast of a2r3 is northwest of a3r3.

east of a3r3 is west of a4r3.

west of a0r1 is east of a5r1.

east of a6r3 is west of a7r3.

east of a8r5 is west of a9r5.

northeast of a4r3 is southwest of a6r3.

east of a9r5 is west of a10r5.

west of a2r3 is east of a11r3.

east of a12r1 is west of a13r1.

north of a14r1 is south of a15r1.

south of a16r1 is north of a17r1.

north of a15r1 is south of a18r1.

south of a1r1 is north of a16r1.

north of a18r1 is south of a19r1.

south of a4r3 is north of a20r3.

south of a3r3 is north of a21r3.

east of a1r1 is west of a19r1.

northeast of a22r2 is southwest of a23r2.

east of a24r4 is west of a25r4.

southwest of a24r4 is northeast of a26r4.

south of a17r1 is north of a28r1.

east of a29r0 is west of a30r0.

northwest of a29r0 is southeast of a31r0.

west of a22r2 is east of a27r2.

northeast of a32r2 is southwest of a33r2.

west of a34r2 is east of a35r2.

south of a36r3 is north of a37r3.

southeast of a38r3 is northwest of a39r3.

northwest of a40r1 is southeast of a41r1.

northwest of a42r3 is southeast of a43r3.

northwest of a44r5 is southeast of a45r5.

northeast of a46r3 is southwest of a47r3.

northeast of a48r5 is southwest of a49r5.

north of a50r0 is south of a51r0.

north of a52r0 is south of a53r0.

northwest of a54r2 is southeast of a55r2.

northwest of a56r4 is southeast of a57r4.

west of a58r1 is east of a59r1.

east of a60r3 is west of a61r3.

west of a62r3 is east of a63r3.

east of a64r1 is west of a65r1.

northeast of a66r1 is southwest of a67r1.

southwest of a68r1 is northeast of a69r1.

southwest of a70r5 is northeast of a71r5.

southeast of a72r5 is northwest of a73r5.

southwest of a74r3 is northeast of a75r3.

southeast of a76r1 is northwest of a77r1.

southeast of a78r3 is northwest of a79r3.

north of a31r0 is south of a80r0.

southwest of a26r4 is northeast of a81r4.

southwest of a29r0 is northeast of a51r0.

south of a30r0 is north of a53r0.

southeast of a22r2 is northwest of a55r2.

southeast of a25r4 is northwest of a57r4.

southwest of a27r2 is northeast of a33r2.

east of a23r2 is west of a35r2.

east of a21r3 is west of a37r3.

west of a20r3 is east of a39r3.

north of a6r3 is south of a63r3.

north of a0r1 is south of a65r1.

northeast of a13r1 is southwest of a69r1.

northeast of a10r5 is southwest of a71r5.

northwest of a8r5 is southeast of a73r5.

northeast of a7r3 is southwest of a75r3.

northwest of a5r1 is southeast of a77r1.

northwest of a11r3 is southeast of a79r3.

north of a12r1 is south of a59r1.

north of a2r3 is south of a61r3.

southwest of a12r1 is northeast of a19r1.

southeast of a13r1 is northwest of a41r1.

southeast of a7r3 is northwest of a43r3.

southeast of a10r5 is northwest of a45r5.

southwest of a11r3 is northeast of a47r3.

southwest of a8r5 is northeast of a49r5.

southwest of a5r1 is northeast of a67r1.

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
the player is in a39r3.

the orientation of the player is 6.
