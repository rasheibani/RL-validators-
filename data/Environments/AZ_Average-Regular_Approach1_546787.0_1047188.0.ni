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

[create a0r0]
a0r0 is a area. "An area (0) in r0".
the printed name of the a0r0 is "Room 0".
Understand "Area 0 in Room 0" as a0r0.
the parent of the a0r0 is r0.
the X of the a0r0 is "547123.9101123596".
the Y of the a0r0 is "1045600.0".

[create a1r0]
a1r0 is a area. "An area (1) in r0".
the printed name of the a1r0 is "Room 0".
Understand "Area 1 in Room 0" as a1r0.
the parent of the a1r0 is r0.
the X of the a1r0 is "547090.9582940117".
the Y of the a1r0 is "1045598.7861894328".

[create a2r0]
a2r0 is a area. "An area (2) in r0".
the printed name of the a2r0 is "Room 0".
Understand "Area 2 in Room 0" as a2r0.
the parent of the a2r0 is r0.
the X of the a2r0 is "548418.0".
the Y of the a2r0 is "1046088.0".

[create a3r0]
a3r0 is a area. "An area (3) in r0".
the printed name of the a3r0 is "Room 0".
Understand "Area 3 in Room 0" as a3r0.
the parent of the a3r0 is r0.
the X of the a3r0 is "548384.1511123561".
the Y of the a3r0 is "1046067.8660087993".

[create a4r0]
a4r0 is a area. "An area (4) in r0".
the printed name of the a4r0 is "Room 0".
Understand "Area 4 in Room 0" as a4r0.
the parent of the a4r0 is r0.
the X of the a4r0 is "548335.0".
the Y of the a4r0 is "1046125.0".

[create a5r0]
a5r0 is a area. "An area (5) in r0".
the printed name of the a5r0 is "Room 0".
Understand "Area 5 in Room 0" as a5r0.
the parent of the a5r0 is r0.
the X of the a5r0 is "548347.5833538158".
the Y of the a5r0 is "1046097.4302261277".

[create a16r0]
a16r0 is a area. "An area (16) in r0".
the printed name of the a16r0 is "Room 0".
Understand "Area 16 in Room 0" as a16r0.
the parent of the a16r0 is r0.
the X of the a16r0 is "545009.0".
the Y of the a16r0 is "1045591.0".

[create a17r0]
a17r0 is a area. "An area (17) in r0".
the printed name of the a17r0 is "Room 0".
Understand "Area 17 in Room 0" as a17r0.
the parent of the a17r0 is r0.
the X of the a17r0 is "545040.2137161086".
the Y of the a17r0 is "1045616.633971292".

[create a18r0]
a18r0 is a area. "An area (18) in r0".
the printed name of the a18r0 is "Room 0".
Understand "Area 18 in Room 0" as a18r0.
the parent of the a18r0 is r0.
the X of the a18r0 is "545768.0".
the Y of the a18r0 is "1045591.0".

[create a19r0]
a19r0 is a area. "An area (19) in r0".
the printed name of the a19r0 is "Room 0".
Understand "Area 19 in Room 0" as a19r0.
the parent of the a19r0 is r0.
the X of the a19r0 is "545736.7862838916".
the Y of the a19r0 is "1045616.633971292".

[create a20r0]
a20r0 is a area. "An area (20) in r0".
the printed name of the a20r0 is "Room 0".
Understand "Area 20 in Room 0" as a20r0.
the parent of the a20r0 is r0.
the X of the a20r0 is "546281.0".
the Y of the a20r0 is "1045591.0".

[create a21r0]
a21r0 is a area. "An area (21) in r0".
the printed name of the a21r0 is "Room 0".
Understand "Area 21 in Room 0" as a21r0.
the parent of the a21r0 is r0.
the X of the a21r0 is "546312.2137161086".
the Y of the a21r0 is "1045616.633971292".

[create a22r0]
a22r0 is a area. "An area (22) in r0".
the printed name of the a22r0 is "Room 0".
Understand "Area 22 in Room 0" as a22r0.
the parent of the a22r0 is r0.
the X of the a22r0 is "545000.0".
the Y of the a22r0 is "1045680.0".

[create a23r0]
a23r0 is a area. "An area (23) in r0".
the printed name of the a23r0 is "Room 0".
Understand "Area 23 in Room 0" as a23r0.
the parent of the a23r0 is r0.
the X of the a23r0 is "545040.2945277712".
the Y of the a23r0 is "1045661.5971994373".

[create a24r0]
a24r0 is a area. "An area (24) in r0".
the printed name of the a24r0 is "Room 0".
Understand "Area 24 in Room 0" as a24r0.
the parent of the a24r0 is r0.
the X of the a24r0 is "545777.0".
the Y of the a24r0 is "1045680.0".

[create a25r0]
a25r0 is a area. "An area (25) in r0".
the printed name of the a25r0 is "Room 0".
Understand "Area 25 in Room 0" as a25r0.
the parent of the a25r0 is r0.
the X of the a25r0 is "545736.7054722288".
the Y of the a25r0 is "1045661.5971994376".

[create a26r0]
a26r0 is a area. "An area (26) in r0".
the printed name of the a26r0 is "Room 0".
Understand "Area 26 in Room 0" as a26r0.
the parent of the a26r0 is r0.
the X of the a26r0 is "546272.0".
the Y of the a26r0 is "1045680.0".

[create a27r0]
a27r0 is a area. "An area (27) in r0".
the printed name of the a27r0 is "Room 0".
Understand "Area 27 in Room 0" as a27r0.
the parent of the a27r0 is r0.
the X of the a27r0 is "546313.3556858187".
the Y of the a27r0 is "1045661.7045075548".

[create a32r0]
a32r0 is a area. "An area (32) in r0".
the printed name of the a32r0 is "Room 0".
Understand "Area 32 in Room 0" as a32r0.
the parent of the a32r0 is r0.
the X of the a32r0 is "548344.0".
the Y of the a32r0 is "1045600.0".

[create a33r0]
a33r0 is a area. "An area (33) in r0".
the printed name of the a33r0 is "Room 0".
Understand "Area 33 in Room 0" as a33r0.
the parent of the a33r0 is r0.
the X of the a33r0 is "548305.8721910112".
the Y of the a33r0 is "1045634.2841014757".

[create a36r0]
a36r0 is a area. "An area (36) in r0".
the printed name of the a36r0 is "Room 0".
Understand "Area 36 in Room 0" as a36r0.
the parent of the a36r0 is r0.
the X of the a36r0 is "545056.3340978593".
the Y of the a36r0 is "1045643.5012742099".

[create a37r0]
a37r0 is a area. "An area (37) in r0".
the printed name of the a37r0 is "Room 0".
Understand "Area 37 in Room 0" as a37r0.
the parent of the a37r0 is r0.
the X of the a37r0 is "545720.6659021409".
the Y of the a37r0 is "1045643.5012742099".

[create a38r0]
a38r0 is a area. "An area (38) in r0".
the printed name of the a38r0 is "Room 0".
Understand "Area 38 in Room 0" as a38r0.
the parent of the a38r0 is r0.
the X of the a38r0 is "546328.634469697".
the Y of the a38r0 is "1045644.0018939393".

[create a39r0]
a39r0 is a area. "An area (39) in r0".
the printed name of the a39r0 is "Room 0".
Understand "Area 39 in Room 0" as a39r0.
the parent of the a39r0 is r0.
the X of the a39r0 is "547083.7656177363".
the Y of the a39r0 is "1045677.3982798846".

[create a40r0]
a40r0 is a area. "An area (40) in r0".
the printed name of the a40r0 is "Room 0".
Understand "Area 40 in Room 0" as a40r0.
the parent of the a40r0 is r0.
the X of the a40r0 is "548363.5885022805".
the Y of the a40r0 is "1046061.5003010029".

[create a42r0]
a42r0 is a area. "An area (42) in r0".
the printed name of the a42r0 is "Room 0".
Understand "Area 42 in Room 0" as a42r0.
the parent of the a42r0 is r0.
the X of the a42r0 is "546813.6310905215".
the Y of the a42r0 is "1045779.0038218704".

[create a44r0]
a44r0 is a area. "An area (44) in r0".
the printed name of the a44r0 is "Room 0".
Understand "Area 44 in Room 0" as a44r0.
the parent of the a44r0 is r0.
the X of the a44r0 is "545604.9376088154".
the Y of the a44r0 is "1046442.9379071897".

[create a45r0]
a45r0 is a area. "An area (45) in r0".
the printed name of the a45r0 is "Room 0".
Understand "Area 45 in Room 0" as a45r0.
the parent of the a45r0 is r0.
the X of the a45r0 is "546472.7066311061".
the Y of the a45r0 is "1046456.8600346774".

[create a48r0]
a48r0 is a area. "An area (48) in r0".
the printed name of the a48r0 is "Room 0".
Understand "Area 48 in Room 0" as a48r0.
the parent of the a48r0 is r0.
the X of the a48r0 is "545383.3961068324".
the Y of the a48r0 is "1045728.733816152".

[create a49r0]
a49r0 is a area. "An area (49) in r0".
the printed name of the a49r0 is "Room 0".
Understand "Area 49 in Room 0" as a49r0.
the parent of the a49r0 is r0.
the X of the a49r0 is "548217.5314997832".
the Y of the a49r0 is "1045731.5748380313".

[create a50r0]
a50r0 is a area. "An area (50) in r0".
the printed name of the a50r0 is "Room 0".
Understand "Area 50 in Room 0" as a50r0.
the parent of the a50r0 is r0.
the X of the a50r0 is "546733.3333727301".
the Y of the a50r0 is "1045788.528606548".

[create a14r1]
a14r1 is a area. "An area (14) in r1".
the printed name of the a14r1 is "Room 1".
Understand "Area 14 in Room 1" as a14r1.
the parent of the a14r1 is r1.
the X of the a14r1 is "545888.0".
the Y of the a14r1 is "1047701.0".

[create a15r1]
a15r1 is a area. "An area (15) in r1".
the printed name of the a15r1 is "Room 1".
Understand "Area 15 in Room 1" as a15r1.
the parent of the a15r1 is r1.
the X of the a15r1 is "545928.1595627206".
the Y of the a15r1 is "1047646.4402178597".

[create a30r1]
a30r1 is a area. "An area (30) in r1".
the printed name of the a30r1 is "Room 1".
Understand "Area 30 in Room 1" as a30r1.
the parent of the a30r1 is r1.
the X of the a30r1 is "546155.0".
the Y of the a30r1 is "1047732.0".

[create a31r1]
a31r1 is a area. "An area (31) in r1".
the printed name of the a31r1 is "Room 1".
Understand "Area 31 in Room 1" as a31r1.
the parent of the a31r1 is r1.
the X of the a31r1 is "546127.2530412681".
the Y of the a31r1 is "1047679.6512252082".

[create a35r1]
a35r1 is a area. "An area (35) in r1".
the printed name of the a35r1 is "Room 1".
Understand "Area 35 in Room 1" as a35r1.
the parent of the a35r1 is r1.
the X of the a35r1 is "546002.5871593202".
the Y of the a35r1 is "1047564.2265159852".

[create a46r1]
a46r1 is a area. "An area (46) in r1".
the printed name of the a46r1 is "Room 1".
Understand "Area 46 in Room 1" as a46r1.
the parent of the a46r1 is r1.
the X of the a46r1 is "546054.717689321".
the Y of the a46r1 is "1047563.3394657804".

[create a6r2]
a6r2 is a area. "An area (6) in r2".
the printed name of the a6r2 is "Room 2".
Understand "Area 6 in Room 2" as a6r2.
the parent of the a6r2 is r2.
the X of the a6r2 is "546704.0".
the Y of the a6r2 is "1047225.0".

[create a7r2]
a7r2 is a area. "An area (7) in r2".
the printed name of the a7r2 is "Room 2".
Understand "Area 7 in Room 2" as a7r2.
the parent of the a7r2 is r2.
the X of the a7r2 is "546741.8974585026".
the Y of the a7r2 is "1047254.2159204248".

[create a8r2]
a8r2 is a area. "An area (8) in r2".
the printed name of the a8r2 is "Room 2".
Understand "Area 8 in Room 2" as a8r2.
the parent of the a8r2 is r2.
the X of the a8r2 is "546787.0".
the Y of the a8r2 is "1047188.0".

[create a9r2]
a9r2 is a area. "An area (9) in r2".
the printed name of the a9r2 is "Room 2".
Understand "Area 9 in Room 2" as a9r2.
the parent of the a9r2 is r2.
the X of the a9r2 is "546778.9921842946".
the Y of the a9r2 is "1047225.8338188234".

[create a10r2]
a10r2 is a area. "An area (10) in r2".
the printed name of the a10r2 is "Room 2".
Understand "Area 10 in Room 2" as a10r2.
the parent of the a10r2 is r2.
the X of the a10r2 is "548369.0".
the Y of the a10r2 is "1047603.0".

[create a11r2]
a11r2 is a area. "An area (11) in r2".
the printed name of the a11r2 is "Room 2".
Understand "Area 11 in Room 2" as a11r2.
the parent of the a11r2 is r2.
the X of the a11r2 is "548305.3734208059".
the Y of the a11r2 is "1047601.0894865774".

[create a12r2]
a12r2 is a area. "An area (12) in r2".
the printed name of the a12r2 is "Room 2".
Understand "Area 12 in Room 2" as a12r2.
the parent of the a12r2 is r2.
the X of the a12r2 is "548314.0".
the Y of the a12r2 is "1047704.0".

[create a13r2]
a13r2 is a area. "An area (13) in r2".
the printed name of the a13r2 is "Room 2".
Understand "Area 13 in Room 2" as a13r2.
the parent of the a13r2 is r2.
the X of the a13r2 is "548280.1666666667".
the Y of the a13r2 is "1047652.8382838285".

[create a28r2]
a28r2 is a area. "An area (28) in r2".
the printed name of the a28r2 is "Room 2".
Understand "Area 28 in Room 2" as a28r2.
the parent of the a28r2 is r2.
the X of the a28r2 is "546741.0".
the Y of the a28r2 is "1047723.0".

[create a29r2]
a29r2 is a area. "An area (29) in r2".
the printed name of the a29r2 is "Room 2".
Understand "Area 29 in Room 2" as a29r2.
the parent of the a29r2 is r2.
the X of the a29r2 is "546770.9449404762".
the Y of the a29r2 is "1047700.0".

[create a34r2]
a34r2 is a area. "An area (34) in r2".
the printed name of the a34r2 is "Room 2".
Understand "Area 34 in Room 2" as a34r2.
the parent of the a34r2 is r2.
the X of the a34r2 is "548257.1617869516".
the Y of the a34r2 is "1047609.4858537383".

[create a41r2]
a41r2 is a area. "An area (41) in r2".
the printed name of the a41r2 is "Room 2".
Understand "Area 41 in Room 2" as a41r2.
the parent of the a41r2 is r2.
the X of the a41r2 is "546754.616203835".
the Y of the a41r2 is "1047258.4234448731".

[create a43r2]
a43r2 is a area. "An area (43) in r2".
the printed name of the a43r2 is "Room 2".
Understand "Area 43 in Room 2" as a43r2.
the parent of the a43r2 is r2.
the X of the a43r2 is "548148.4980734903".
the Y of the a43r2 is "1047557.3206812767".

[create a47r2]
a47r2 is a area. "An area (47) in r2".
the printed name of the a47r2 is "Room 2".
Understand "Area 47 in Room 2" as a47r2.
the parent of the a47r2 is r2.
the X of the a47r2 is "546870.4414637962".
the Y of the a47r2 is "1047581.5150352007".

[create d0]
d0 is a door. "A door between a42r0 and a43r2".
d0 is northeast of a42r0 and southwest of a43r2.

[create d2]
d2 is a door. "A door between a45r0 and a46r1".
d2 is north of a45r0 and south of a46r1.

[create d3]
d3 is a door. "A door between a44r0 and a35r1".
d3 is north of a44r0 and south of a35r1.

west of a0r0 is east of a1r0.

southwest of a2r0 is northeast of a3r0.

southeast of a4r0 is northwest of a5r0.

northeast of a6r2 is southwest of a7r2.

north of a8r2 is south of a9r2.

west of a10r2 is east of a11r2.

southwest of a12r2 is northeast of a13r2.

southeast of a14r1 is northwest of a15r1.

northeast of a16r0 is southwest of a17r0.

northwest of a18r0 is southeast of a19r0.

northeast of a20r0 is southwest of a21r0.

southeast of a22r0 is northwest of a23r0.

southwest of a24r0 is northeast of a25r0.

southeast of a26r0 is northwest of a27r0.

southeast of a28r2 is northwest of a29r2.

southwest of a30r1 is northeast of a31r1.

northwest of a32r0 is southeast of a33r0.

west of a11r2 is east of a34r2.

southwest of a13r2 is northeast of a34r2.

southeast of a15r1 is northwest of a35r1.

southeast of a23r0 is northwest of a36r0.

southwest of a25r0 is northeast of a37r0.

southeast of a27r0 is northwest of a38r0.

north of a1r0 is south of a39r0.

west of a3r0 is east of a40r0.

southeast of a5r0 is northwest of a40r0.

east of a7r2 is west of a41r2.

northwest of a9r2 is southeast of a41r2.

northeast of a17r0 is southwest of a36r0.

northwest of a19r0 is southeast of a37r0.

northeast of a21r0 is southwest of a38r0.

east of a44r0 is west of a45r0.

north of a41r2 is south of a47r2.

southwest of a34r2 is northeast of a43r2.

west of a43r2 is east of a47r2.

southeast of a29r2 is northwest of a47r2.

east of a35r1 is west of a46r1.

west of a37r0 is east of a48r0.

west of a39r0 is east of a42r0.

east of a39r0 is west of a49r0.

northwest of a33r0 is southeast of a49r0.

southwest of a40r0 is northeast of a49r0.

south of a44r0 is north of a48r0.

west of a42r0 is east of a50r0.

south of a45r0 is north of a50r0.

east of a38r0 is west of a50r0.

east of a36r0 is west of a48r0.

southwest of a31r1 is northeast of a46r1.

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
the player is in a8r2.

the orientation of the player is 6.
