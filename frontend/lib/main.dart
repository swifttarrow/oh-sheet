import 'package:flutter/material.dart';

import 'api/client.dart';
import 'screens/upload_screen.dart';
import 'theme.dart';

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
      theme: OhSheetTheme.light,
      home: _AppShell(api: _api),
    );
  }
}

class _AppShell extends StatefulWidget {
  const _AppShell({required this.api});
  final OhSheetApi api;

  @override
  State<_AppShell> createState() => _AppShellState();
}

class _AppShellState extends State<_AppShell> {
  int _currentIndex = 0;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(
        index: _currentIndex,
        children: [
          UploadScreen(api: widget.api),
          const _LibraryPlaceholder(),
          const _ProfilePlaceholder(),
        ],
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _currentIndex,
        onDestinationSelected: (i) => setState(() => _currentIndex = i),
        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.home_outlined),
            selectedIcon: Icon(Icons.home),
            label: 'Home',
          ),
          NavigationDestination(
            icon: Icon(Icons.library_music_outlined),
            selectedIcon: Icon(Icons.library_music),
            label: 'Library',
          ),
          NavigationDestination(
            icon: Icon(Icons.person_outline),
            selectedIcon: Icon(Icons.person),
            label: 'Profile',
          ),
        ],
      ),
    );
  }
}

class _LibraryPlaceholder extends StatelessWidget {
  const _LibraryPlaceholder();

  @override
  Widget build(BuildContext context) {
    return const Scaffold(
      body: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.library_music, size: 64, color: OhSheetColors.mutedText),
            SizedBox(height: 16),
            Text(
              'Community Library',
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.w600),
            ),
            SizedBox(height: 8),
            Text('Coming soon', style: TextStyle(color: OhSheetColors.mutedText)),
          ],
        ),
      ),
    );
  }
}

class _ProfilePlaceholder extends StatelessWidget {
  const _ProfilePlaceholder();

  @override
  Widget build(BuildContext context) {
    return const Scaffold(
      body: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.person, size: 64, color: OhSheetColors.mutedText),
            SizedBox(height: 16),
            Text(
              'Profile',
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.w600),
            ),
            SizedBox(height: 8),
            Text('Coming soon', style: TextStyle(color: OhSheetColors.mutedText)),
          ],
        ),
      ),
    );
  }
}
