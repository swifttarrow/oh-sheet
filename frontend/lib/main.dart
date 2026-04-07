import 'package:flutter/material.dart';

import 'api/client.dart';
import 'responsive.dart';
import 'screens/upload_screen.dart';
import 'theme.dart';
import 'widgets/sticker_widgets.dart';

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
    return LayoutBuilder(
      builder: (context, constraints) {
        final wide = constraints.maxWidth >= OhSheetBreakpoints.sideNav;
        final pages = [
          UploadScreen(api: widget.api),
          const _LibraryPlaceholder(),
          const _ProfilePlaceholder(),
        ];

        if (wide) {
          return Scaffold(
            backgroundColor: OhSheetColors.cream,
            body: Row(
              children: [
                DecoratedBox(
                  decoration: BoxDecoration(
                    color: Colors.white,
                    border: Border(
                      right: BorderSide(color: OhSheetColors.inkStroke, width: 2.5),
                    ),
                    boxShadow: [
                      BoxShadow(
                        color: OhSheetColors.inkStroke.withValues(alpha: 0.07),
                        offset: const Offset(4, 0),
                        blurRadius: 14,
                      ),
                    ],
                  ),
                  child: NavigationRail(
                    selectedIndex: _currentIndex,
                    onDestinationSelected: (i) => setState(() => _currentIndex = i),
                    labelType: NavigationRailLabelType.all,
                    destinations: const [
                      NavigationRailDestination(
                        icon: Icon(Icons.home_outlined),
                        selectedIcon: Icon(Icons.home),
                        label: Text('Home'),
                      ),
                      NavigationRailDestination(
                        icon: Icon(Icons.library_music_outlined),
                        selectedIcon: Icon(Icons.library_music),
                        label: Text('Library'),
                      ),
                      NavigationRailDestination(
                        icon: Icon(Icons.person_outline),
                        selectedIcon: Icon(Icons.person),
                        label: Text('Profile'),
                      ),
                    ],
                  ),
                ),
                Expanded(
                  child: IndexedStack(
                    index: _currentIndex,
                    children: pages,
                  ),
                ),
              ],
            ),
          );
        }

        return Scaffold(
          backgroundColor: OhSheetColors.cream,
          body: IndexedStack(
            index: _currentIndex,
            children: pages,
          ),
          bottomNavigationBar: Padding(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 14),
            child: DecoratedBox(
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(28),
                border: Border.all(color: OhSheetColors.inkStroke, width: 2.5),
                boxShadow: OhSheetStickerStyle.stickerShadows,
              ),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(25),
                child: NavigationBar(
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
              ),
            ),
          ),
        );
      },
    );
  }
}

class _LibraryPlaceholder extends StatelessWidget {
  const _LibraryPlaceholder();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: OhSheetColors.cream,
      body: const OhSheetResponsiveBody(
        maxWidth: 420,
        alignTop: false,
        padding: EdgeInsets.all(24),
        child: OhSheetSticker(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(Icons.library_music, size: 64, color: OhSheetColors.teal),
              SizedBox(height: 16),
              Text(
                'Community Library',
                style: TextStyle(fontSize: 20, fontWeight: FontWeight.w800),
              ),
              SizedBox(height: 8),
              Text(
                'Coming soon — browse everyone’s sheets here.',
                textAlign: TextAlign.center,
                style: TextStyle(color: OhSheetColors.mutedText, fontWeight: FontWeight.w600),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _ProfilePlaceholder extends StatelessWidget {
  const _ProfilePlaceholder();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: OhSheetColors.cream,
      body: const OhSheetResponsiveBody(
        maxWidth: 420,
        alignTop: false,
        padding: EdgeInsets.all(24),
        child: OhSheetSticker(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(Icons.person, size: 64, color: OhSheetColors.orange),
              SizedBox(height: 16),
              Text(
                'Profile',
                style: TextStyle(fontSize: 20, fontWeight: FontWeight.w800),
              ),
              SizedBox(height: 8),
              Text(
                'Coming soon — your account & prefs live here.',
                textAlign: TextAlign.center,
                style: TextStyle(color: OhSheetColors.mutedText, fontWeight: FontWeight.w600),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
