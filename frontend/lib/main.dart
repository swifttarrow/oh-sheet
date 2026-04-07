import 'package:flutter/material.dart';

import 'api/client.dart';
import 'screens/upload_screen.dart';

void main() {
  runApp(const OhSheetApp());
}

class OhSheetApp extends StatefulWidget {
  const OhSheetApp({super.key});

  @override
  State<OhSheetApp> createState() => _OhSheetAppState();
}

class _OhSheetAppState extends State<OhSheetApp> {
  final OhSheetApi _api = OhSheetApi();

  @override
  void dispose() {
    _api.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Oh Sheet',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: UploadScreen(api: _api),
    );
  }
}
