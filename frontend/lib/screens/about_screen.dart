library;

import 'package:flutter/material.dart';

import '../responsive.dart';
import '../theme.dart';
import '../widgets/sticker_widgets.dart';

class AboutScreen extends StatelessWidget {
  const AboutScreen({super.key});

  static const List<_TeamMember> _members = [
    _TeamMember(
      name: 'Jack Jiang',
      blurb:
          'Senior full-stack engineer with 8 years of experience across product development, SRE, and workflow automation. Meme enthusiast and climbing addict.',
      assetPath: 'assets/about/jack_jiang.png',
      accent: OhSheetColors.orange,
      surfaceTint: Color(0xFFFFFCF6),
      tilt: -0.030,
      imageAlignment: Alignment(0, 0.28),
      imageAspectRatio: 1.0,
      frameScale: 1.0,
      imageScale: 1.0,
    ),
    _TeamMember(
      name: 'Luis Ramos',
      blurb:
          'Senior software engineer who architects enterprise solutions across stacks, platforms, and frameworks with an AI-first approach to development.',
      assetPath: 'assets/about/luis_ramos.jpg',
      accent: OhSheetColors.tealBright,
      surfaceTint: Color(0xFFF5FFFD),
      tilt: 0.022,
      imageAlignment: Alignment.topCenter,
      imageAspectRatio: 1.0,
      frameScale: 1.0,
      imageScale: 1.0,
    ),
    _TeamMember(
      name: 'Raq Robinson',
      blurb:
          'Senior full-stack engineer who blends React frontend expertise with backend and systems thinking to build scalable, secure, and accessible platforms. Recently led high-impact work across marketing applications, crypto R&D, and modernized financial tooling at MassMutual.',
      assetPath: 'assets/about/raq_robinson.jpg',
      accent: OhSheetColors.pinkAccent,
      surfaceTint: Color(0xFFFFF7FB),
      tilt: -0.015,
      imageAlignment: Alignment.topCenter,
      imageAspectRatio: 1.0,
      frameScale: 1.0,
      imageScale: 1.0,
    ),
    _TeamMember(
      name: 'Kevin Chang',
      blurb:
          'Senior full-stack engineer with 13 years of startup experience building products end to end. Piano wizard, board game lover, and badminton junkie.',
      assetPath: 'assets/about/kevin_chang.png',
      accent: OhSheetColors.teal,
      surfaceTint: Color(0xFFF6FBFF),
      tilt: 0.024,
      imageAlignment: Alignment.center,
      imageAspectRatio: 1.0,
      frameScale: 1.0,
      imageScale: 1.0,
    ),
    _TeamMember(
      name: 'Ross Kuehl',
      blurb:
          'Senior backend-focused software engineer with deep expertise in Python, cloud infrastructure, and distributed systems. Has built and scaled high-performance APIs and microservices, modernized legacy systems, and led engineering best practices across startups and production environments.',
      assetPath: 'assets/about/ross_kuehl.png',
      accent: OhSheetColors.orange,
      surfaceTint: Color(0xFFFFFBF3),
      tilt: -0.022,
      imageAlignment: Alignment.center,
      imageAspectRatio: 1.0,
      frameScale: 1.0,
      imageScale: 1.0,
    ),
  ];

  @override
  Widget build(BuildContext context) {
    return const Scaffold(
      backgroundColor: OhSheetColors.cream,
      body: SafeArea(
        child: SingleChildScrollView(
          child: OhSheetResponsiveBody(
            maxWidth: 1080,
            padding: EdgeInsets.fromLTRB(16, 20, 16, 28),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                _AboutHero(),
                SizedBox(height: 22),
                OhSheetStickerSectionTitle(
                  text: 'Meet the team',
                  accent: OhSheetColors.teal,
                ),
                SizedBox(height: 14),
                _TeamMosaic(members: _members),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _AboutHero extends StatelessWidget {
  const _AboutHero();

  @override
  Widget build(BuildContext context) {
    return OhSheetSticker(
      backgroundColor: const Color(0xFFFFFDF8),
      padding: const EdgeInsets.fromLTRB(24, 24, 24, 24),
      child: Stack(
        clipBehavior: Clip.none,
        children: [
          Positioned(
            top: -14,
            right: 18,
            child: _AccentBubble(
              color: OhSheetColors.orange.withValues(alpha: 0.17),
              size: 94,
            ),
          ),
          Positioned(
            left: 18,
            bottom: -20,
            child: _AccentBubble(
              color: OhSheetColors.teal.withValues(alpha: 0.14),
              size: 72,
            ),
          ),
          LayoutBuilder(
            builder: (context, constraints) {
              final stacked = constraints.maxWidth < 820;
              final intro = Column(
                crossAxisAlignment: stacked
                    ? CrossAxisAlignment.center
                    : CrossAxisAlignment.start,
                children: [
                  Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                    decoration: BoxDecoration(
                      color: const Color(0xFFFFF0D9),
                      borderRadius: BorderRadius.circular(999),
                      border: Border.all(
                        color: OhSheetColors.inkStroke,
                        width: 2,
                      ),
                    ),
                    child: const Text(
                      'About Oh Sheet!',
                      style: TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.w800,
                        color: OhSheetColors.darkText,
                      ),
                    ),
                  ),
                  const SizedBox(height: 16),
                  const Text(
                    'Five engineers, two weeks, one capstone: Oh Sheet.',
                    textAlign: TextAlign.left,
                    style: TextStyle(
                      fontSize: 30,
                      height: 1.12,
                      fontWeight: FontWeight.w900,
                      color: OhSheetColors.darkText,
                    ),
                  ),
                  const SizedBox(height: 10),
                  Text(
                    'The team came together through the GauntletAI program and built Oh Sheet as their capstone project. In just two weeks, they turned the idea into a working product that transforms songs into playable piano sheet music.',
                    textAlign: stacked ? TextAlign.center : TextAlign.left,
                    style: const TextStyle(
                      fontSize: 15,
                      height: 1.45,
                      color: OhSheetColors.mutedText,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  const SizedBox(height: 18),
                  Wrap(
                    alignment:
                        stacked ? WrapAlignment.center : WrapAlignment.start,
                    spacing: 10,
                    runSpacing: 10,
                    children: const [
                      _HeroBadge(
                        icon: Icons.groups_2_outlined,
                        label: '5 engineers',
                      ),
                      _HeroBadge(
                        icon: Icons.school_outlined,
                        label: 'GauntletAI capstone',
                      ),
                      _HeroBadge(
                        icon: Icons.schedule_outlined,
                        label: 'Built in 2 weeks',
                      ),
                    ],
                  ),
                ],
              );

              const callout = _HeroCallout();

              if (stacked) {
                return Column(
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: [
                    intro,
                    const SizedBox(height: 18),
                    callout,
                  ],
                );
              }

              return Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Expanded(flex: 7, child: intro),
                  const SizedBox(width: 20),
                  const Expanded(flex: 4, child: _HeroCallout()),
                ],
              );
            },
          ),
        ],
      ),
    );
  }
}

class _HeroCallout extends StatelessWidget {
  const _HeroCallout();

  @override
  Widget build(BuildContext context) {
    return const OhSheetSticker(
      backgroundColor: Color(0xFFEFFFFA),
      padding: EdgeInsets.all(18),
      radius: 22,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.auto_awesome_rounded, color: OhSheetColors.orange),
              SizedBox(width: 8),
              Expanded(
                child: Text(
                  'From cohort to capstone',
                  style: TextStyle(
                    fontSize: 17,
                    fontWeight: FontWeight.w800,
                    color: OhSheetColors.darkText,
                  ),
                ),
              ),
            ],
          ),
          SizedBox(height: 10),
          Text(
            'Oh Sheet came together through focused collaboration across product, frontend, backend, and infrastructure disciplines during the GauntletAI program.',
            style: TextStyle(
              fontSize: 14,
              height: 1.45,
              color: OhSheetColors.mutedText,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }
}

class _HeroBadge extends StatelessWidget {
  const _HeroBadge({required this.icon, required this.label});

  final IconData icon;
  final String label;

  @override
  Widget build(BuildContext context) {
    return DecoratedBox(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: OhSheetColors.inkStroke, width: 2),
      ),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 16, color: OhSheetColors.teal),
            const SizedBox(width: 7),
            Text(
              label,
              style: const TextStyle(
                color: OhSheetColors.darkText,
                fontWeight: FontWeight.w700,
                fontSize: 13,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _AccentBubble extends StatelessWidget {
  const _AccentBubble({required this.color, required this.size});

  final Color color;
  final double size;

  @override
  Widget build(BuildContext context) {
    return IgnorePointer(
      child: Container(
        width: size,
        height: size,
        decoration: BoxDecoration(
          color: color,
          shape: BoxShape.circle,
          border: Border.all(
            color: OhSheetColors.inkStroke.withValues(alpha: 0.15),
            width: 2,
          ),
        ),
      ),
    );
  }
}

class _TeamMosaic extends StatelessWidget {
  const _TeamMosaic({required this.members});

  final List<_TeamMember> members;

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        const gap = 18.0;
        if (constraints.maxWidth >= 980) {
          final cardWidth = (constraints.maxWidth - (gap * 2)) / 3;
          final secondRowWidth = (cardWidth * 2) + gap;
          return Column(
            children: [
              Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Expanded(child: _MemberCard(member: members[0])),
                  const SizedBox(width: gap),
                  Expanded(
                    child: Padding(
                      padding: const EdgeInsets.only(top: 24),
                      child: _MemberCard(member: members[1]),
                    ),
                  ),
                  const SizedBox(width: gap),
                  Expanded(
                    child: Padding(
                      padding: const EdgeInsets.only(top: 6),
                      child: _MemberCard(member: members[2]),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 18),
              Align(
                alignment: Alignment.topCenter,
                child: SizedBox(
                  width: secondRowWidth,
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Expanded(
                        child: Padding(
                          padding: const EdgeInsets.only(top: 2),
                          child: _MemberCard(member: members[3]),
                        ),
                      ),
                      const SizedBox(width: gap),
                      Expanded(
                        child: Padding(
                          padding: const EdgeInsets.only(top: 10),
                          child: _MemberCard(member: members[4]),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ],
          );
        }

        if (constraints.maxWidth >= 640) {
          return Column(
            children: [
              _MemberCard(member: members[0]),
              const SizedBox(height: 16),
              Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Expanded(child: _MemberCard(member: members[1])),
                  const SizedBox(width: 16),
                  Expanded(child: _MemberCard(member: members[2])),
                ],
              ),
              const SizedBox(height: 16),
              Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Expanded(child: _MemberCard(member: members[3])),
                  const SizedBox(width: 16),
                  Expanded(child: _MemberCard(member: members[4])),
                ],
              ),
            ],
          );
        }

        return Column(
          children: [
            for (var i = 0; i < members.length; i++) ...[
              _MemberCard(member: members[i]),
              if (i != members.length - 1) const SizedBox(height: 16),
            ],
          ],
        );
      },
    );
  }
}

class _MemberCard extends StatelessWidget {
  const _MemberCard({required this.member});

  final _TeamMember member;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(4),
      child: Transform.rotate(
        angle: member.tilt,
        child: OhSheetSticker(
          backgroundColor: member.surfaceTint,
          padding: const EdgeInsets.all(14),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Align(
                alignment: Alignment.topCenter,
                child: FractionallySizedBox(
                  widthFactor: member.frameScale,
                  child: DecoratedBox(
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(20),
                      border: Border.all(
                        color: OhSheetColors.inkStroke,
                        width: 2.5,
                      ),
                      boxShadow: [
                        BoxShadow(
                          color:
                              OhSheetColors.inkStroke.withValues(alpha: 0.08),
                          offset: const Offset(0, 4),
                          blurRadius: 12,
                        ),
                      ],
                    ),
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(17),
                      child: AspectRatio(
                        aspectRatio: member.imageAspectRatio,
                        child: ColoredBox(
                          color: Colors.white,
                          child: Center(
                            child: FractionallySizedBox(
                              widthFactor: member.imageScale,
                              heightFactor: member.imageScale,
                              child: Image.asset(
                                member.assetPath,
                                fit: BoxFit.cover,
                                alignment: member.imageAlignment,
                                errorBuilder: (context, error, stackTrace) {
                                  return ColoredBox(
                                    color: Colors.white,
                                    child: Center(
                                      child: Icon(
                                        Icons.groups_2_outlined,
                                        size: 48,
                                        color: member.accent,
                                      ),
                                    ),
                                  );
                                },
                              ),
                            ),
                          ),
                        ),
                      ),
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 14),
              Row(
                children: [
                  Container(
                    width: 12,
                    height: 12,
                    decoration: BoxDecoration(
                      color: member.accent,
                      shape: BoxShape.circle,
                      border: Border.all(
                        color: OhSheetColors.inkStroke,
                        width: 1.5,
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      member.name,
                      style: const TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.w800,
                        color: OhSheetColors.darkText,
                      ),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 10),
              Text(
                member.blurb,
                style: const TextStyle(
                  fontSize: 13.5,
                  height: 1.45,
                  color: OhSheetColors.mutedText,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _TeamMember {
  const _TeamMember({
    required this.name,
    required this.blurb,
    required this.assetPath,
    required this.accent,
    required this.surfaceTint,
    required this.tilt,
    required this.imageAlignment,
    required this.imageAspectRatio,
    required this.frameScale,
    required this.imageScale,
  });

  final String name;
  final String blurb;
  final String assetPath;
  final Color accent;
  final Color surfaceTint;
  final double tilt;
  final Alignment imageAlignment;
  final double imageAspectRatio;
  final double frameScale;
  final double imageScale;
}
