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

[create a18r0]
a18r0 is a area. "An area (18) in r0".
the printed name of the a18r0 is "Room 0".
Understand "Area 18 in Room 0" as a18r0.
the parent of the a18r0 is r0.
the X of the a18r0 is "545370.3987047074".
the Y of the a18r0 is "2153494.3342435663".

[create a19r0]
a19r0 is a area. "An area (19) in r0".
the printed name of the a19r0 is "Room 0".
Understand "Area 19 in Room 0" as a19r0.
the parent of the a19r0 is r0.
the X of the a19r0 is "545479.6250000001".
the Y of the a19r0 is "2153602.5".

[create a26r0]
a26r0 is a area. "An area (26) in r0".
the printed name of the a26r0 is "Room 0".
Understand "Area 26 in Room 0" as a26r0.
the parent of the a26r0 is r0.
the X of the a26r0 is "545261.3750000002".
the Y of the a26r0 is "2153601.379310345".

[create a33r0]
a33r0 is a area. "An area (33) in r0".
the printed name of the a33r0 is "Room 0".
Understand "Area 33 in Room 0" as a33r0.
the parent of the a33r0 is r0.
the X of the a33r0 is "545516.0".
the Y of the a33r0 is "2153640.0".

[create a39r0]
a39r0 is a area. "An area (39) in r0".
the printed name of the a39r0 is "Room 0".
Understand "Area 39 in Room 0" as a39r0.
the parent of the a39r0 is r0.
the X of the a39r0 is "545225.0".
the Y of the a39r0 is "2153640.0".

[create a2r1]
a2r1 is a area. "An area (2) in r1".
the printed name of the a2r1 is "Room 1".
Understand "Area 2 in Room 1" as a2r1.
the parent of the a2r1 is r1.
the X of the a2r1 is "545261.375".
the Y of the a2r1 is "2151438.620689655".

[create a3r1]
a3r1 is a area. "An area (3) in r1".
the printed name of the a3r1 is "Room 1".
Understand "Area 3 in Room 1" as a3r1.
the parent of the a3r1 is r1.
the X of the a3r1 is "545370.4748096998".
the Y of the a3r1 is "2151545.7374361246".

[create a4r1]
a4r1 is a area. "An area (4) in r1".
the printed name of the a4r1 is "Room 1".
Understand "Area 4 in Room 1" as a4r1.
the parent of the a4r1 is r1.
the X of the a4r1 is "545479.625".
the Y of the a4r1 is "2151438.4".

[create a21r1]
a21r1 is a area. "An area (21) in r1".
the printed name of the a21r1 is "Room 1".
Understand "Area 21 in Room 1" as a21r1.
the parent of the a21r1 is r1.
the X of the a21r1 is "545418.8902277271".
the Y of the a21r1 is "2152346.087573426".

[create a44r1]
a44r1 is a area. "An area (44) in r1".
the printed name of the a44r1 is "Room 1".
Understand "Area 44 in Room 1" as a44r1.
the parent of the a44r1 is r1.
the X of the a44r1 is "545225.0".
the Y of the a44r1 is "2151400.0".

[create a45r1]
a45r1 is a area. "An area (45) in r1".
the printed name of the a45r1 is "Room 1".
Understand "Area 45 in Room 1" as a45r1.
the parent of the a45r1 is r1.
the X of the a45r1 is "545516.0".
the Y of the a45r1 is "2151400.0".

[create a13r2]
a13r2 is a area. "An area (13) in r2".
the printed name of the a13r2 is "Room 2".
Understand "Area 13 in Room 2" as a13r2.
the parent of the a13r2 is r2.
the X of the a13r2 is "546979.5536306141".
the Y of the a13r2 is "2152807.8868319127".

[create a22r2]
a22r2 is a area. "An area (22) in r2".
the printed name of the a22r2 is "Room 2".
Understand "Area 22 in Room 2" as a22r2.
the parent of the a22r2 is r2.
the X of the a22r2 is "545917.9961837009".
the Y of the a22r2 is "2152692.273223205".

[create a23r2]
a23r2 is a area. "An area (23) in r2".
the printed name of the a23r2 is "Room 2".
Understand "Area 23 in Room 2" as a23r2.
the parent of the a23r2 is r2.
the X of the a23r2 is "546666.5433955578".
the Y of the a23r2 is "2153498.2236823137".

[create a24r2]
a24r2 is a area. "An area (24) in r2".
the printed name of the a24r2 is "Room 2".
Understand "Area 24 in Room 2" as a24r2.
the parent of the a24r2 is r2.
the X of the a24r2 is "546895.4699720407".
the Y of the a24r2 is "2153457.948467067".

[create a25r2]
a25r2 is a area. "An area (25) in r2".
the printed name of the a25r2 is "Room 2".
Understand "Area 25 in Room 2" as a25r2.
the parent of the a25r2 is r2.
the X of the a25r2 is "547033.0".
the Y of the a25r2 is "2153599.2457710593".

[create a31r2]
a31r2 is a area. "An area (31) in r2".
the printed name of the a31r2 is "Room 2".
Understand "Area 31 in Room 2" as a31r2.
the parent of the a31r2 is r2.
the X of the a31r2 is "546608.0".
the Y of the a31r2 is "2153640.0".

[create a32r2]
a32r2 is a area. "An area (32) in r2".
the printed name of the a32r2 is "Room 2".
Understand "Area 32 in Room 2" as a32r2.
the parent of the a32r2 is r2.
the X of the a32r2 is "546639.2499999995".
the Y of the a32r2 is "2153555.042466668".

[create a38r2]
a38r2 is a area. "An area (38) in r2".
the printed name of the a38r2 is "Room 2".
Understand "Area 38 in Room 2" as a38r2.
the parent of the a38r2 is r2.
the X of the a38r2 is "547042.0".
the Y of the a38r2 is "2153640.0".

[create a9r3]
a9r3 is a area. "An area (9) in r3".
the printed name of the a9r3 is "Room 3".
Understand "Area 9 in Room 3" as a9r3.
the parent of the a9r3 is r3.
the X of the a9r3 is "547364.125".
the Y of the a9r3 is "2151448.507644491".

[create a10r3]
a10r3 is a area. "An area (10) in r3".
the printed name of the a10r3 is "Room 3".
Understand "Area 10 in Room 3" as a10r3.
the parent of the a10r3 is r3.
the X of the a10r3 is "547445.3622708634".
the Y of the a10r3 is "2151553.5109334085".

[create a11r3]
a11r3 is a area. "An area (11) in r3".
the printed name of the a11r3 is "Room 3".
Understand "Area 11 in Room 3" as a11r3.
the parent of the a11r3 is r3.
the X of the a11r3 is "547506.9412801261".
the Y of the a11r3 is "2151554.610558574".

[create a12r3]
a12r3 is a area. "An area (12) in r3".
the printed name of the a12r3 is "Room 3".
Understand "Area 12 in Room 3" as a12r3.
the parent of the a12r3 is r3.
the X of the a12r3 is "547586.875".
the Y of the a12r3 is "2151450.0994746266".

[create a35r3]
a35r3 is a area. "An area (35) in r3".
the printed name of the a35r3 is "Room 3".
Understand "Area 35 in Room 3" as a35r3.
the parent of the a35r3 is r3.
the X of the a35r3 is "547327.0".
the Y of the a35r3 is "2151400.0".

[create a36r3]
a36r3 is a area. "An area (36) in r3".
the printed name of the a36r3 is "Room 3".
Understand "Area 36 in Room 3" as a36r3.
the parent of the a36r3 is r3.
the X of the a36r3 is "547624.0".
the Y of the a36r3 is "2151400.0".

[create a14r4]
a14r4 is a area. "An area (14) in r4".
the printed name of the a14r4 is "Room 4".
Understand "Area 14 in Room 4" as a14r4.
the parent of the a14r4 is r4.
the X of the a14r4 is "548093.556510393".
the Y of the a14r4 is "2153502.53788561".

[create a15r4]
a15r4 is a area. "An area (15) in r4".
the printed name of the a15r4 is "Room 4".
Understand "Area 15 in Room 4" as a15r4.
the parent of the a15r4 is r4.
the X of the a15r4 is "548229.8305982074".
the Y of the a15r4 is "2153498.8947038357".

[create a27r4]
a27r4 is a area. "An area (27) in r4".
the printed name of the a27r4 is "Room 4".
Understand "Area 27 in Room 4" as a27r4.
the parent of the a27r4 is r4.
the X of the a27r4 is "548302.9000000001".
the Y of the a27r4 is "2153589.466754617".

[create a28r4]
a28r4 is a area. "An area (28) in r4".
the printed name of the a28r4 is "Room 4".
Understand "Area 28 in Room 4" as a28r4.
the parent of the a28r4 is r4.
the X of the a28r4 is "548022.0999999996".
the Y of the a28r4 is "2153589.523978247".

[create a40r4]
a40r4 is a area. "An area (40) in r4".
the printed name of the a40r4 is "Room 4".
Understand "Area 40 in Room 4" as a40r4.
the parent of the a40r4 is r4.
the X of the a40r4 is "548338.0".
the Y of the a40r4 is "2153640.0".

[create a41r4]
a41r4 is a area. "An area (41) in r4".
the printed name of the a41r4 is "Room 4".
Understand "Area 41 in Room 4" as a41r4.
the parent of the a41r4 is r4.
the X of the a41r4 is "547987.0".
the Y of the a41r4 is "2153640.0".

[create a5r5]
a5r5 is a area. "An area (5) in r5".
the printed name of the a5r5 is "Room 5".
Understand "Area 5 in Room 5" as a5r5.
the parent of the a5r5 is r5.
the X of the a5r5 is "548852.0751328205".
the Y of the a5r5 is "2151554.3386380887".

[create a6r5]
a6r5 is a area. "An area (6) in r5".
the printed name of the a6r5 is "Room 5".
Understand "Area 6 in Room 5" as a6r5.
the parent of the a6r5 is r5.
the X of the a6r5 is "548933.0".
the Y of the a6r5 is "2151449.4606324406".

[create a7r5]
a7r5 is a area. "An area (7) in r5".
the printed name of the a7r5 is "Room 5".
Understand "Area 7 in Room 5" as a7r5.
the parent of the a7r5 is r5.
the X of the a7r5 is "548723.0".
the Y of the a7r5 is "2151449.4226234336".

[create a8r5]
a8r5 is a area. "An area (8) in r5".
the printed name of the a8r5 is "Room 5".
Understand "Area 8 in Room 5" as a8r5.
the parent of the a8r5 is r5.
the X of the a8r5 is "548803.2576810808".
the Y of the a8r5 is "2151553.7046452095".

[create a46r5]
a46r5 is a area. "An area (46) in r5".
the printed name of the a46r5 is "Room 5".
Understand "Area 46 in Room 5" as a46r5.
the parent of the a46r5 is r5.
the X of the a46r5 is "548968.0".
the Y of the a46r5 is "2151400.0".

[create a47r5]
a47r5 is a area. "An area (47) in r5".
the printed name of the a47r5 is "Room 5".
Understand "Area 47 in Room 5" as a47r5.
the parent of the a47r5 is r5.
the X of the a47r5 is "548688.0".
the Y of the a47r5 is "2151400.0".

[create a16r6]
a16r6 is a area. "An area (16) in r6".
the printed name of the a16r6 is "Room 6".
Understand "Area 16 in Room 6" as a16r6.
the parent of the a16r6 is r6.
the X of the a16r6 is "549315.4999999997".
the Y of the a16r6 is "2153591.129062261".

[create a17r6]
a17r6 is a area. "An area (17) in r6".
the printed name of the a17r6 is "Room 6".
Understand "Area 17 in Room 6" as a17r6.
the parent of the a17r6 is r6.
the X of the a17r6 is "549389.0274643078".
the Y of the a17r6 is "2153502.368728805".

[create a20r6]
a20r6 is a area. "An area (20) in r6".
the printed name of the a20r6 is "Room 6".
Understand "Area 20 in Room 6" as a20r6.
the parent of the a20r6 is r6.
the X of the a20r6 is "549534.5000000005".
the Y of the a20r6 is "2153609.786912203".

[create a34r6]
a34r6 is a area. "An area (34) in r6".
the printed name of the a34r6 is "Room 6".
Understand "Area 34 in Room 6" as a34r6.
the parent of the a34r6 is r6.
the X of the a34r6 is "549571.0".
the Y of the a34r6 is "2153640.0".

[create a42r6]
a42r6 is a area. "An area (42) in r6".
the printed name of the a42r6 is "Room 6".
Understand "Area 42 in Room 6" as a42r6.
the parent of the a42r6 is r6.
the X of the a42r6 is "549279.0".
the Y of the a42r6 is "2153640.0".

[create a0r7]
a0r7 is a area. "An area (0) in r7".
the printed name of the a0r7 is "Room 7".
Understand "Area 0 in Room 7" as a0r7.
the parent of the a0r7 is r7.
the X of the a0r7 is "546806.6768153389".
the Y of the a0r7 is "2151619.9404216".

[create a1r7]
a1r7 is a area. "An area (1) in r7".
the printed name of the a1r7 is "Room 7".
Understand "Area 1 in Room 7" as a1r7.
the parent of the a1r7 is r7.
the X of the a1r7 is "546987.625".
the Y of the a1r7 is "2151413.09086189".

[create a29r7]
a29r7 is a area. "An area (29) in r7".
the printed name of the a29r7 is "Room 7".
Understand "Area 29 in Room 7" as a29r7.
the parent of the a29r7 is r7.
the X of the a29r7 is "546696.0".
the Y of the a29r7 is "2151472.0815910874".

[create a30r7]
a30r7 is a area. "An area (30) in r7".
the printed name of the a30r7 is "Room 7".
Understand "Area 30 in Room 7" as a30r7.
the parent of the a30r7 is r7.
the X of the a30r7 is "546770.921189243".
the Y of the a30r7 is "2151613.851069941".

[create a37r7]
a37r7 is a area. "An area (37) in r7".
the printed name of the a37r7 is "Room 7".
Understand "Area 37 in Room 7" as a37r7.
the parent of the a37r7 is r7.
the X of the a37r7 is "546733.0".
the Y of the a37r7 is "2151400.0".

[create a43r7]
a43r7 is a area. "An area (43) in r7".
the printed name of the a43r7 is "Room 7".
Understand "Area 43 in Room 7" as a43r7.
the parent of the a43r7 is r7.
the X of the a43r7 is "547024.0".
the Y of the a43r7 is "2151426.18172378".

[create d0]
d0 is a door. "A door between a13r2 and a10r3".
d0 is south of a13r2 and north of a10r3.

[create d2]
d2 is a door. "A door between a15r4 and a8r5".
d2 is south of a15r4 and north of a8r5.

[create d4]
d4 is a door. "A door between a5r5 and a17r6".
d4 is north of a5r5 and south of a17r6.

[create d6]
d6 is a door. "A door between a11r3 and a14r4".
d6 is north of a11r3 and south of a14r4.

[create d8]
d8 is a door. "A door between a22r2 and a30r7".
d8 is southeast of a22r2 and northwest of a30r7.

[create d9]
d9 is a door. "A door between a13r2 and a0r7".
d9 is southwest of a13r2 and northeast of a0r7.

[create d12]
d12 is a door. "A door between a21r1 and a22r2".
d12 is northeast of a21r1 and southwest of a22r2.

[create d14]
d14 is a door. "A door between a18r0 and a21r1".
d14 is south of a18r0 and north of a21r1.

southeast of a0r7 is northwest of a1r7.

northeast of a2r1 is southwest of a3r1.

southeast of a3r1 is northwest of a4r1.

southeast of a5r5 is northwest of a6r5.

northeast of a7r5 is southwest of a8r5.

northeast of a9r3 is southwest of a10r3.

southeast of a11r3 is northwest of a12r3.

east of a14r4 is west of a15r4.

southeast of a16r6 is northwest of a17r6.

northeast of a18r0 is southwest of a19r0.

northeast of a17r6 is southwest of a20r6.

northeast of a22r2 is southwest of a23r2.

north of a13r2 is south of a24r2.

northeast of a24r2 is southwest of a25r2.

east of a23r2 is west of a24r2.

northwest of a18r0 is southeast of a26r0.

northeast of a15r4 is southwest of a27r4.

northwest of a14r4 is southeast of a28r4.

north of a3r1 is south of a21r1.

northeast of a29r7 is southwest of a30r7.

south of a31r2 is north of a32r2.

northeast of a19r0 is southwest of a33r0.

northeast of a20r6 is southwest of a34r6.

southwest of a9r3 is northeast of a35r3.

southeast of a12r3 is northwest of a36r3.

southeast of a29r7 is northwest of a37r7.

north of a25r2 is south of a38r2.

northwest of a26r0 is southeast of a39r0.

northeast of a27r4 is southwest of a40r4.

northwest of a28r4 is southeast of a41r4.

northwest of a16r6 is southeast of a42r6.

east of a1r7 is west of a43r7.

southwest of a2r1 is northeast of a44r1.

southeast of a4r1 is northwest of a45r1.

southeast of a6r5 is northwest of a46r5.

southwest of a7r5 is northeast of a47r5.

west of a5r5 is east of a8r5.

east of a10r3 is west of a11r3.

west of a0r7 is east of a30r7.

northwest of a23r2 is southeast of a32r2.

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
the player is in a13r2.

the orientation of the player is 6.
